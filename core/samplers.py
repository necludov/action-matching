import numpy as np
import torch


class RWMH:
    def __init__(self, target, sigma=5.0):
        self.device = target.device
        self.target = target
        self.sigma = sigma
    
    @torch.no_grad()
    def log_prob(self, x):
        return self.target.log_prob(x).view([-1,1])
        
    def sample_n(self, x_0, n):
        # x_0.shape = [batch_size, dim]
        x = x_0.clone()
        log_p = self.log_prob(x)
        ar = torch.zeros_like(log_p)
        for i in range(n):
            _x = x + self.sigma*torch.empty_like(x).normal_()
            _log_p = self.log_prob(_x)
            accept_mask = (_log_p - log_p > torch.log(torch.zeros_like(log_p).uniform_())).float()
            x = accept_mask*_x + (1-accept_mask)*x
            log_p = accept_mask*_log_p + (1-accept_mask)*log_p
            ar += accept_mask
        ar = ar/n
        return x, ar
