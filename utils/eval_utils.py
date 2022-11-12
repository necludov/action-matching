import torch
import math
import numpy as np

from scipy import integrate


def euler_scheme(ode_func, t0, t1, x, dt):
    solution = dotdict()
    timesteps = np.arange(t0, t1, dt)
    solution.y = np.zeros([len(x), len(timesteps)+1])
    solution.y[:,0] = x
    solution.nfev = 0
    for i, t in enumerate(timesteps):
        dx = ode_func(t, solution.y[:,i])
        solution.y[:,i+1] = solution.y[:,i] + dt*dx
        solution.nfev += 1
    return solution
    
@torch.no_grad()
def solve_ode(device, dxdt, x, t0=1.0, t1=0.0, atol=1e-5, rtol=1e-5, method='RK45', dt=-1e-2):
    shape = x.shape
    def ode_func(t, x_):
        x_ = torch.from_numpy(x_).reshape(shape).to(device).type(torch.float32)
        t_vec = torch.ones(x_.shape[0], device=x_.device) * t
        with torch.enable_grad():
            x_.requires_grad = True
            dx = dxdt(t_vec,x_).detach()
            x_.requires_grad = False
        dx = dx.cpu().numpy().flatten()
        return dx
    
    x = x.detach().cpu().numpy().flatten()
    if 'euler' != method:
        solution = integrate.solve_ivp(ode_func, (t0, t1), x, rtol=rtol, atol=atol, method=method)
    else:
        solution = euler_scheme(ode_func, t0, t1, x, dt)
    return torch.from_numpy(solution.y[:,-1].reshape(shape)), solution.nfev

@torch.no_grad()
def get_likelihood(device, dxdt, x, t0=0.0, t1=1.0, atol=1e-5, rtol=1e-5, method='RK45', dt=1e-2):
    assert (2 == x.dim())
    shape = x.shape
    eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
    x = x.detach().cpu().numpy().flatten()

    def ode_func(t, x_):
        x_ = torch.from_numpy(x_[:-shape[0]]).reshape(shape).to(device).type(torch.float32)
        t_vec = torch.ones(x_.shape[0], device=x_.device) * t
        with torch.enable_grad():
            x_.requires_grad = True
            dx = dxdt(t_vec,x_)
            div = (eps*torch.autograd.grad(dx, x_, grad_outputs=eps)[0]).sum(1)
            x_.requires_grad = False
        dx = dx.detach().cpu().numpy().flatten()
        div = div.detach().cpu().numpy().flatten()
        return np.concatenate([dx, div], axis=0)

    init = np.concatenate([x, np.zeros((shape[0],))], axis=0)
    if 'euler' != method:
        solution = integrate.solve_ivp(ode_func, (t0, t1), init, rtol=rtol, atol=atol, method=method)
    else:
        solution = euler_scheme(ode_func, t0, t1, init, dt)
    
    z = torch.from_numpy(solution.y[:-shape[0],-1]).reshape(shape).to(device).type(torch.float32)
    delta_logp = torch.from_numpy(solution.y[-shape[0]:,-1]).to(device).type(torch.float32)
    return delta_logp, z, solution.nfev

######################################################
# Code from https://github.com/josipd/torch-two-sample
######################################################

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

    def pval(self, distances, n_permutations=1000):
        r"""Compute a p-value using a permutation test.
        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.MMDStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.
        Returns
        -------
        float
            The estimated p-value."""
        if isinstance(distances, Variable):
            distances = distances.data
        return permutation_test_mat(distances.cpu().numpy(),
                                    self.n_1, self.n_2,
                                    n_permutations,
                                    a00=self.a00, a11=self.a11, a01=self.a01)

