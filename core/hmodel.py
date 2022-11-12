import torch
import numpy as np

from scipy.special import genlaguerre
from scipy.special import lpmv
from scipy.special import factorial2


hbar = 1.0 # reduced Planck constant
m_e = 1.0 # electron mass
eps0 = 1.0
e = 1.0
a0 = 4*np.pi*eps0*hbar**2/m_e/e**2 # Bohr radius
EPS = 1e-2

def asslaguerre_torch(n, alpha, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 1.0 + alpha - x
    else:
        output = (2*n-1.0+alpha-x)*asslaguerre_torch(n-1,alpha,x)
        output = output-(n-1+alpha)*asslaguerre_torch(n-2,alpha,x)
        output = output/n
        return output
    
def asslegendre_torch(m, l, x):
    if m < 0:
        m = np.abs(m)
        return (-1)**m*np.math.factorial(l-m)/np.math.factorial(l+m)*asslegendre_torch(m, l, x)
    if m == l:
        return (-1)**l*factorial2(2*l-1)*torch.pow(1.0-x**2, torch.tensor([l]).to(x.device)/2.0)
    elif m > l:
        return torch.zeros_like(x)
    else:
        output = (2*l-1)*x*asslegendre_torch(m, l-1, x) 
        output = output - (l+m-1)*asslegendre_torch(m, l-2, x)
        output = output/(l-m)
        return output

class EigenState:
    def __init__(self, n, l, m):
        self.n, self.l, self.m = n, l, m
        self.E = -hbar**2/(2.0*m_e*a0**2)/self.n**2
        self.L2 = hbar**2*self.l*(self.l+1)
        self.Lz = hbar**2*self.m
        
    def radial(self, r):
        n, l, m = self.n, self.l, self.m
#         output = torch.exp(-r/n/a0)*(2*r/(n*a0))**l
        output = torch.exp(-r/n/a0 + l*torch.log(2*r) - l*np.log(n*a0))
        output = output*np.sqrt((2.0/n/a0)**3*np.math.factorial(n-l-1)/np.math.factorial(n+l)/2.0/n)
        output = output*asslaguerre_torch(n-l-1, 2*l+1, 2.0*r/(n*a0))
        return output
    
    def angular(self, theta, phi):
        n, l, m = self.n, self.l, self.m
        output = asslegendre_torch(m, l, torch.cos(theta))
        output = output*(-1)**m*np.sqrt((2.0*l+1.0)*np.math.factorial(l-m)/np.math.factorial(l+m)/4.0/np.pi)
        output = output*torch.exp(1.0j*m*phi)
        return output
    
    def _radial(self, r):
        n, l, m = self.n, self.l, self.m
        output = np.sqrt((2.0/n/a0)**3*np.math.factorial(n-l-1)/np.math.factorial(n+l)/2.0/n)
        output = output*asslaguerre_torch(n-l-1, 2*l+1, 2.0*r/(n*a0))
        return output
    
    def _radial_log(self, r):
        n, l, m = self.n, self.l, self.m
        return -r/n/a0 + l*torch.log(2*r) - l*np.log(n*a0)
    
class WaveFunction:
    def __init__(self, n, l, m, c0, device):
        assert (n < 0).sum() == 0
        assert (l < 0).sum() == (l >= n).sum() == 0
        assert (m > l).sum() == (m < -l).sum() == 0
        self.n, self.l, self.m, self.c0 = n, l, m, c0
        self.c0 = self.c0.to(device)
        self.c0 = self.c0/torch.sqrt(torch.sum(self.c0.abs()**2))
        self.states = list(EigenState(qnum[0], qnum[1], qnum[2]) for qnum in zip(n,l,m))
        self.dim = 3
        self.device = device
        
    def evolve_to(self, t):
        E = torch.tensor(list(psi.E for psi in self.states)).to(self.device)
        return WaveFunction(self.n, self.l, self.m, torch.exp(-1j*E*t/hbar)*self.c0, self.device)
    
    def avgH(self):
        E = torch.tensor(list(psi.E for psi in self.states))
        return torch.sum(self.c0.abs()**2*E)
    
    def avgL2(self):
        L2 = torch.tensor(list(psi.L2 for psi in self.states))
        return torch.sum(self.c0.abs()**2*L2)
    
    def avgLz(self):
        Lz = torch.tensor(list(psi.Lz for psi in self.states))
        return torch.sum(self.c0.abs()**2*Lz)
    
    def at(self, x):
        r = torch.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2+EPS).flatten()
        theta = torch.atan2(torch.sqrt(x[:,0]**2 + x[:,1]**2+EPS),x[:,2]).flatten()
        x_coord = torch.sign(x[:,0])*(torch.abs(x[:,0])+EPS)
        phi = torch.atan2(x[:,1],x_coord).flatten()
        return self.at_polar(r, theta, phi)
        
    def at_polar(self, r, theta, phi):
        assert r.shape == theta.shape == phi.shape
        output = 1j*torch.zeros_like(r)
        for i in range(len(self.states)):
            psi_i = self.states[i]
            output += self.c0[i]*psi_i.radial(r)*psi_i.angular(theta, phi)
        return output
    
    def log_prob(self, x):
        # x.shape = [num_points, 3]
        r = torch.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2+EPS).flatten()
        z = torch.sign(x[:,2])*(torch.abs(x[:,2])+EPS)
        theta = torch.atan2(torch.sqrt(x[:,0]**2 + x[:,1]**2+EPS),z).flatten()
        x_coord = torch.sign(x[:,0])*(torch.abs(x[:,0])+EPS)
        phi = torch.atan2(x[:,1],x_coord).flatten()
        
        radial_log = torch.stack([psi._radial_log(r) for psi in self.states])
        angular = torch.stack([psi._radial(r)*psi.angular(theta, phi) for psi in self.states])
        coords = self.c0.view([-1,1])
        max_log, _ = torch.max(radial_log, dim=0)
        psi = (torch.exp(radial_log-max_log)*angular*coords).sum(0)
        output = 2*torch.log(psi.abs()) + 2*max_log
        return output

    
class BohmianDynamics:
    def __init__(self, wave_function, samples):
        self.psi = wave_function
        self.samples = samples
        
    def propagate(self, dt):
        samples = self.samples
        samples.requires_grad = True
        v = torch.autograd.grad(self.psi.at(samples).angle().sum(), samples)[0]
        samples.data += dt*v
        samples.requires_grad = False
        self.samples = samples
        self.psi = self.psi.evolve_to(dt)
