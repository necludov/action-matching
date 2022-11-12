import torch
import torch.nn as nn
import math

class ActionNet(nn.Module):
    def __init__(self, net):
        super(ActionNet, self).__init__()
        self.net = net
    
    def forward(self, t, x):
        h = self.net(t, x)
        out = 0.5*((x-h)**2).sum(dim=1, keepdim=True)
        return out
    
    def propagate(self, t, x, dt):
        x.requires_grad = True
        v = torch.autograd.grad(self(t,x).sum(), x)[0]
        x.data += dt*v
        x.requires_grad = False
        return x


class ScoreNet(nn.Module):
    def __init__(self, net):
        super(ScoreNet, self).__init__()
        self.net = net
    
    def forward(self, t, x):
        return 1e-2*self.net(t, x)
    
    def propagate(self, t, x, dt, eps=15.0, n_steps=5):
        for _ in range(n_steps):
            x.data += 0.5*eps*self(t+dt, x) + math.sqrt(eps)*torch.randn_like(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_dims=4, n_out=3, n_hid=512, layer=nn.Linear, relu=False):
        super(MLP, self).__init__()
        self._built = False
        self.net = nn.Sequential(
            layer(n_dims, n_hid),
            nn.LeakyReLU(.2) if relu else nn.SiLU(n_hid),
            layer(n_hid, n_hid),
            nn.LeakyReLU(.2) if relu else nn.SiLU(n_hid),
            layer(n_hid, n_hid),
            nn.LeakyReLU(.2) if relu else nn.SiLU(n_hid),
            layer(n_hid, n_hid),
            nn.LeakyReLU(.2) if relu else nn.SiLU(n_hid),
            layer(n_hid, n_out)
        )

    def forward(self, t, x):
        x = x.view(x.size(0), -1)
        t = t.view(t.size(0), 1)
        h = torch.hstack([t,x])
        h = self.net(h)
        return h
