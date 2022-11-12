import torch
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns

def plot_scatter(samples, R=200, bins=75, axes=None):
    data = samples.detach().cpu().numpy()
    
    fig = None
    fs = 15
    if axes is None:
        fig, axes = plt.subplots(1,3)
    order = [[0,1], [1,2], [0,2]]
    labels = ['x', 'y', 'z']
    for i in range(len(axes)):
        a = axes[i]
        current_a = order[i]
        a.set_xlabel(labels[current_a[0]], fontsize=fs)
        a.set_ylabel(labels[current_a[1]], fontsize=fs)
        a.set_xlim(-R,R)
        a.set_ylim(-R,R)
        a.scatter(data[:,current_a[0]], data[:,current_a[1]], cmap = cm.jet, linewidth=0, marker=".")
    return fig

def plot_samples_kde(samples, R=200, ppa=300, axes=None):
    data = samples.detach().cpu().numpy()
    
    fig = None
    fs = 15
    if axes is None:
        fig, axes = plt.subplots(1,3)
    order = [[0,1], [1,2], [0,2]]
    labels = ['x', 'y', 'z']
    for i in range(len(axes)):
        a = axes[i]
        current_a = order[i]
        a.set_xlabel(labels[current_a[0]], fontsize=fs)
        a.set_ylabel(labels[current_a[1]], fontsize=fs)
        a.set_xlim(-R,R)
        a.set_ylim(-R,R)
        sns.kdeplot(x=data[:,current_a[0]], y=data[:,current_a[1]], 
                    fill=True, thresh=0, levels=40, cmap=cm.jet, ax=axes[0], bw_adjust=6e-1)
    return fig

def plot_samples(samples, R=200, bins=50, axes=None):
    data = samples.detach().cpu().numpy()
    
    fig = None
    fs = 15
    if axes is None:
        fig, axes = plt.subplots(1,3)
    order = [[0,1], [1,2], [0,2]]
    labels = ['x', 'y', 'z']
    for i in range(len(axes)):
        a = axes[i]
        current_a = order[i]
        a.set_xlabel(labels[current_a[0]], fontsize=fs)
        a.set_ylabel(labels[current_a[1]], fontsize=fs)
        a.set_xlim(-R,R)
        a.set_ylim(-R,R)
        a.hist2d(data[:,current_a[0]], data[:,current_a[1]], cmap = cm.jet, 
                 bins=bins, range=[[-R,R],[-R,R]])
    return fig

def plot_projections(prob, R=200, ppa=300, device=torch.device('cpu'), axes=None):
    x, y, z = torch.linspace(-R, R, ppa), torch.linspace(-R, R, ppa), torch.linspace(-R, R, ppa)
    x, y, z = torch.meshgrid(x, y, z, indexing='xy')
    x, y, z = x.to(device), y.to(device), z.to(device)
    
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    probs = prob(points).reshape([ppa,ppa,ppa]).detach().cpu().numpy().transpose([1,0,2])

    fig = None
    fs = 15
    if axes is None:
        fig, axes = plt.subplots(1,3)
    order = [[0,1], [1,2], [0,2]]
    third_a = [2, 0, 1]
    labels = ['x', 'y', 'z']
    for i in range(len(axes)):
        a = axes[i]
        current_a = order[i]
        a.set_xlabel(labels[current_a[0]], fontsize=fs)
        a.set_ylabel(labels[current_a[1]], fontsize=fs)
        a.set_xlim(-R,R)
        a.set_ylim(-R,R)
        a.imshow(probs.sum(third_a[i]).T, cmap = cm.jet, origin='lower', extent = [-R,R,-R,R])
    return fig
    
def plot_slices(prob, R=300, ppa=300, device=torch.device('cpu')):
    x, y = torch.linspace(-R, R, ppa), torch.linspace(-R, R, ppa)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.to(device)
    y = y.to(device)

    fs = 15
    plt.subplot(131)
    prob_vals = prob(torch.stack([x.flatten(), y.flatten(), torch.zeros_like(x.flatten())], 1)).detach().cpu().numpy()
    plt.imshow(prob_vals.reshape(x.shape), cmap = cm.jet, origin='lower', extent = [-R,R,-R,R])
    plt.xlabel('x', fontsize=fs)
    plt.ylabel('y', fontsize=fs)
    plt.subplot(132)
    prob_vals = prob(torch.stack([torch.zeros_like(x.flatten()), x.flatten(), y.flatten()], 1)).detach().cpu().numpy()
    plt.imshow(prob_vals.reshape(x.shape), cmap = cm.jet, origin='lower', extent = [-R,R,-R,R])
    plt.xlabel('y', fontsize=fs)
    plt.ylabel('z', fontsize=fs)
    plt.subplot(133)
    prob_vals = prob(torch.stack([x.flatten(), torch.zeros_like(x.flatten()), y.flatten()], 1)).detach().cpu().numpy()
    plt.imshow(prob_vals.reshape(x.shape), cmap = cm.jet, origin='lower', extent = [-R,R,-R,R])
    plt.xlabel('x', fontsize=fs)
    plt.ylabel('z', fontsize=fs)
    
def plot_sphere(prob, ppa=300):
    phi = torch.linspace(0.0, 2*np.pi, ppa)
    theta = torch.linspace(0.0, 2*np.pi, ppa)
    phi, theta = torch.meshgrid(phi, theta, indexing='xy')
    phi, theta, r = phi.flatten(), theta.flatten(), torch.ones_like(theta.flatten())
    x = torch.stack([r*torch.cos(phi)*torch.sin(theta), r*torch.sin(phi)*torch.sin(theta), r*torch.cos(theta)], 1)
    rho = prob(x).reshape([ppa, ppa])
    rho, xs, ys, zs = rho.numpy(), x[:,0].numpy(), x[:,1].numpy(), x[:,2].numpy()
    xs, ys, zs = xs.reshape([ppa, ppa]), ys.reshape([ppa, ppa]), zs.reshape([ppa, ppa])
    color_map = ListedColormap(sns.color_palette("coolwarm", 50))
    scalarMap = cm.ScalarMappable(norm = plt.Normalize(vmin=np.min(rho), vmax =np.max(rho)), 
                                  cmap = cm.jet)
    C = scalarMap.to_rgba(rho)
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(121, projection ='3d')
    ax1.plot_surface(xs, ys, zs, rcount = 100, ccount = 100, color = 'b', facecolors = C)
    plt.xlabel('x')
    plt.ylabel('y')

    ax2 = fig.add_subplot(122, projection ='3d')
    ax2.plot_surface(rho*xs, rho*ys, rho*zs, rstride = 1, cstride = 1, color = 'b', facecolors = C)
    plt.xlabel('x')
    plt.ylabel('y')
