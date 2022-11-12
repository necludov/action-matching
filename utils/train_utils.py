import torch
import numpy as np

from core.samplers import RWMH
from core.hmodel import *
from core.losses import *
from models.mlp import *
from utils.eval_utils import *
from tqdm.auto import tqdm, trange

import wandb


class DataGenerator:
    def __init__(self, psi, config):
        self.sampler = RWMH(psi, sigma=10.0)
        self.bs = config.data.batch_size
        self.samples, self.times = None, None
        self.n_steps = config.data.n_steps # in time
        self.T = config.data.T
        self.psi = psi
        self.device = self.psi.device
        self.gen_data()
        
    def gen_data(self):
        batch_size, T, n_steps = self.bs, self.T, self.n_steps
        psi = self.psi
        x_0 = 50*torch.zeros([batch_size, psi.dim], device=psi.device).normal_()
        x, ar = self.sampler.sample_n(x_0, 5000)
        dynamics = BohmianDynamics(psi, x)
        dt = T/n_steps
        samples = torch.zeros([n_steps+1, batch_size, psi.dim], device=torch.device('cpu'))
        samples[0] = dynamics.samples.cpu()
        times = torch.zeros(n_steps + 1, device=torch.device('cpu'))
        for i in range(n_steps):
            dynamics.propagate(dt)
            samples[i+1] = dynamics.samples.cpu()
            times[i+1] = times[i] + dt
        self.samples, self.times = samples, times
        
    def q_t(self, t, replace=False):
        device = t.device
        t_ids = torch.round(t*(len(self.samples)-1)).long().squeeze().cpu()
        sample_ids = torch.from_numpy(np.random.choice(self.bs, t.shape[0], replace=replace)).squeeze()
        subsamples = self.samples[t_ids, sample_ids, :].to(device)
        times = self.times[t_ids].reshape(-1,1).to(device)
        return subsamples, times/self.T

    
def prepare_hydrogen(device, config=None):
    if config is None:
        from configs.hydrogen import get_config
        config = get_config()
    n, l, m, c = config.data.n, config.data.l, config.data.m, config.data.c
    psi = WaveFunction(n,l,m,c,device)
    data_gen = DataGenerator(psi, config)
    
    net = MLP(n_hid=config.model.n_hid)
    if 'sm' == config.model.method:
        model = ScoreNet(net)
        loss = SMLoss(model, data_gen.q_t, config, device, sliced=False)
    elif 'ssm' == config.model.method:
        model = ScoreNet(net)
        loss = SMLoss(model, data_gen.q_t, config, device, sliced=True)
    elif 'am' == config.model.method:
        model = ActionNet(net)
        loss = AMLoss(model, data_gen.q_t, config, device)
    else:
        raise NameError(f'undefined method: {config.model.method}')
    model.to(device)
    return model, data_gen, loss, config

def train(model, ema, loss, data_gen, optim, config, device):
    for current_step in range(config.train.n_iter):
        config.train.current_step = current_step
        model.train()
        loss_total = loss.eval_loss()
        optim.zero_grad(set_to_none=True)
        loss_total.backward()

        if config.train.warmup > 0:
            for g in optim.param_groups:
                g['lr'] = config.train.lr * np.minimum(current_step / config.train.warmup, 1.0)
        if config.train.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
        optim.step()
        ema.update(model.parameters())
        
        if (current_step % 50) == 0:
            logging_dict = {
                'loss_' + config.model.method: loss_total.detach().cpu()
            }
            wandb.log(logging_dict, step=current_step)
        
        if ((current_step % config.train.regen_every) == 0) and current_step > 0:
            data_gen.gen_data()
        if ((current_step % config.train.eval_every) == 0):
            metric_dict = evaluate(model, ema, data_gen, device, config)
            wandb.log(metric_dict, step=current_step)
        if ((current_step % config.train.save_every) == 0):
            save(model, ema, optim, loss, config)
    config.train.current_step = current_step
    save(model, ema, optim, loss, config)
    metric_dict = evaluate(model, ema, data_gen, device, config)
    wandb.log(metric_dict, step=current_step)
            
def save(model, ema, optim, loss, config):
    checkpoint_name = config.model.savepath + '_%d.cpt' % config.train.current_step
    config.model.checkpoints.append(checkpoint_name)
    torch.save({'model': model.state_dict(), 
                'ema': ema.state_dict(), 
                'optim': optim.state_dict()}, checkpoint_name)
    torch.save(config, config.model.savepath + '.config')
    
def evaluate(model, ema, data_gen, device, config):
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()
    ######## evaluation ########
    metric_dict = {}
    x = data_gen.samples[0].to(device)
    dt = 1./config.data.n_steps
    t = torch.zeros([x.shape[0],1], device=device)
    n_evals = 10
    eval_every = config.data.n_steps//n_evals
    metric_dict['avg_mmd'] = 0.0
    metric_dict['score_loss'] = 0.0
    mmd = MMDStatistic(config.data.batch_size, config.data.batch_size)
    for i in range(config.data.n_steps):
        x = model.propagate(t, x, dt)
        t.data += dt
        if ((i+1) % eval_every) == 0:
            x_t, gen_t = data_gen.q_t(t, replace=False)
            gen_t = gen_t*data_gen.T
            cur_mmd = mmd(x, x_t, 1e-4*torch.ones(x.shape[1], device=device))
            metric_dict['avg_mmd'] += cur_mmd.abs().cpu().numpy()/n_evals
            if config.model.method in {'sm', 'ssm'}:
                psi_t = data_gen.psi.evolve_to(gen_t[0].to(device))
                x_t.requires_grad = True
                nabla_logp = torch.autograd.grad(psi_t.log_prob(x_t.to(device)).sum(), x_t)[0]
                x_t.requires_grad = False
                loss = 0.5*((nabla_logp - model(t, x_t))**2).sum(1)
                metric_dict['score_loss'] += loss.mean()/n_evals
    ############################
    ema.restore(model.parameters())
    return metric_dict
