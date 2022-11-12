import os
import copy
import argparse
import json

import torch
import numpy as np
import wandb

from torch import nn
from tqdm.auto import tqdm, trange

from models.mlp import *
from models import ema
from utils.train_utils import *

from core.hmodel import *
from core.losses import *
    

def main(args):
    device = torch.device('cuda')
    
    model, data_gen, loss, config = prepare_hydrogen(device)
    config.model.savepath = os.path.join(args.checkpoint_dir, config.model.savepath)
    config.train.wandbid = wandb.util.generate_id()
    
    wandb.login()
    wandb.init(id=config.train.wandbid, 
               project=config.data.name, 
               resume="allow",
               config=json.loads(config.to_json_best_effort()))
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = config.train.wandbid
    
    optim = torch.optim.Adam(model.parameters(), lr=config.train.lr, betas=config.train.betas, 
                             eps=1e-8, weight_decay=config.train.wd)
    ema_ = ema.ExponentialMovingAverage(model.parameters(), decay=config.eval.ema)
    train(model, ema_, loss, data_gen, optim, config, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description=''
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='path to save and look for the checkpoint file',
        default=os.getcwd()
    )
    
    main(parser.parse_args())
