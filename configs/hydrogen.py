from ml_collections import config_dict
import torch
import numpy as np

def get_config():
    config = config_dict.ConfigDict()
    
    config.data = config_dict.ConfigDict()
    config.data.T = 14_000
    config.data.n_steps = 1_000
    config.data.batch_size = 5_000
    config.data.n = torch.tensor([3,2])
    config.data.l = torch.tensor([2,1])
    config.data.m = torch.tensor([-1,0])
    config.data.c = torch.tensor([1.0+0.0j, 1.0+0.0j])
    config.data.name = 'hydrogen'
    
    config.model = config_dict.ConfigDict()
    config.model.method = 'am'
    config.model.n_hid = 256
    config.model.savepath = config.model.method + '_' + config.data.name
    config.model.checkpoints = []

    config.train = config_dict.ConfigDict()
    config.train.batch_size = 1_000
    config.train.lr = 1e-4
    config.train.warmup = 5_000
    config.train.grad_clip = 1.0
    config.train.betas = (0.9, 0.999)
    config.train.wd = 0.0
    config.train.n_iter = 100_000
    config.train.regen_every = 5_000
    config.train.save_every = 10_000
    config.train.eval_every = 5_000
    config.train.current_step = 0
    
    config.eval = config_dict.ConfigDict()
    config.eval.ema = 0.9999
    
    return config
