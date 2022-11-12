## Action Matching

### Schrödinger Equation Simulation

We demonstrate that Action Matching can learn a wide range of stochastic dynamics by applying it to the dynamics of a quantum system evolving according to the Schrödinger equation. The Schrödinger equation describes the evolution of many quantum systems, and in particular, it describes the physics of molecular systems. Here, for the ground truth dynamics, we take the dynamics of an excited state of the hydrogen atom, which is described by the following equation

$i\frac{\partial}{\partial t}\psi(x,t) = -\frac{1}{|x|}\psi(x,t) -\frac{1}{2}\nabla^2\psi(x,t).$

The function $\psi(x,t): \mathbb{R}^3\times \mathbb{R} \to \mathbb{C}$ is called a wavefunction and it completely describes the state of the quantum system.
In particular, it defines the distribution of the coordinates $x$ by defining its density as $q_t(x) := |\psi(x,t)|^2$, which dynamics is defined by the dynamics of $\psi(x,t)$.
Below we demonstrate the ground truth dynamics where we project the density $q_t(x)$ on three different planes.

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/dynamics_densities.gif" alt="drawing" width="800"/>

In what follows we illustrate the histograms for the learned dynamics. Since the original distribution is in $\mathbb{R}^3$ we project the samples onto three different planes and draw 2d-histograms. 
The top row for every model corresponds to the ground truth dynamics (the training data), and the bottom rows corresspond to the learned models.

#### Action Matching (AM) results visualization
<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_results.gif" alt="drawing" width="800"/>

#### Score Matching (SM) results visualization
<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/sm_results.gif" alt="drawing" width="800"/>

#### Sliced Score Matching (SSM) results visualization
<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/ssm_results.gif" alt="drawing" width="800"/>

### Generative Modeling Experiments

CelebA inpainting (left), superresolution (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_inpaint.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_superres.gif" alt="drawing" width="400"/>

CelebA generation on torus (left), via diffusion (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_torus.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_diffusion.gif" alt="drawing" width="400"/>

CIFAR-10 colorization (left), superresolution (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_color.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_superres.gif" alt="drawing" width="400"/>

CIFAR-10 generation on torus (left), via diffusion (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_torus.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_diffusion.gif" alt="drawing" width="400"/>
