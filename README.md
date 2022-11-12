## Action Matching

### Schrödinger Equation Simulation

We demonstrate that Action Matching can learn a wide range of stochastic dynamics by applying it to the dynamics of a quantum system evolving according to the Schrödinger equation. The Schrödinger equation describes the evolution of many quantum systems, and in particular, it describes the physics of molecular systems. Here, for the ground truth dynamics, we take the dynamics of an excited state of the hydrogen atom, which is described by the following equation

$i\frac{\partial}{\partial t}\psi(x,t) = -\frac{1}{|x|}\psi(x,t) -\frac{1}{2}\nabla^2\psi(x,t).$

The function $\psi(x,t): \mathbb{R}^3\times \mathbb{R} \to \mathbb{C}$ is called a wavefunction and it completely describes the state of the quantum system.
In particular, it defines the distribution of the coordinates $x$ by defining its density as $q_t(x) := |\psi(x,t)|^2$, which dynamics is defined by the dynamics of $\psi(x,t)$.

### Generative Modeling Experiments

CelebA inpainting (left), superresolution (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_inpaint.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_superres.gif" alt="drawing" width="400"/>

CelebA generation on torus (left), via diffusion (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_torus.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_celeba_diffusion.gif" alt="drawing" width="400"/>

CIFAR-10 colorization (left), superresolution (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_color.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_superres.gif" alt="drawing" width="400"/>

CIFAR-10 generation on torus (left), via diffusion (right)

<img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_torus.gif" alt="drawing" width="400"/><img src="https://github.com/action-matching/action-matching/blob/main/notebooks/gifs/am_cifar_diffusion.gif" alt="drawing" width="400"/>
