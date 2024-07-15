r"""
Tools for finite difference frequency-domain (FDFD) simulations and calculations.

These mostly involve picking a single frequency, then setting up and solving a
matrix equation (Ax=b) or eigenvalue problem.


Submodules:

- `operators`, `functional`: General FDFD problem setup.
- `solvers`: Solver interface and reference implementation.
- `scpml`: Stretched-coordinate perfectly matched layer (scpml) boundary conditions
- `waveguide_2d`: Operators and mode-solver for waveguides with constant cross-section.
- `waveguide_3d`: Functions for transforming `waveguide_2d` results into 3D.


================================================================

From the "Frequency domain" section of `meanas.fdmath`, we have

$$
 \begin{aligned}
 \tilde{E}_{l, \vec{r}} &= \tilde{E}_{\vec{r}} e^{-\imath \omega l \Delta_t} \\
 \tilde{H}_{l - \frac{1}{2}, \vec{r} + \frac{1}{2}} &= \tilde{H}_{\vec{r} + \frac{1}{2}} e^{-\imath \omega (l - \frac{1}{2}) \Delta_t} \\
 \tilde{J}_{l, \vec{r}} &= \tilde{J}_{\vec{r}} e^{-\imath \omega (l - \frac{1}{2}) \Delta_t} \\
 \tilde{M}_{l - \frac{1}{2}, \vec{r} + \frac{1}{2}} &= \tilde{M}_{\vec{r} + \frac{1}{2}} e^{-\imath \omega l \Delta_t} \\
 \hat{\nabla} \times (\mu^{-1}_{\vec{r} + \frac{1}{2}} \cdot \tilde{\nabla} \times \tilde{E}_{\vec{r}})
    -\Omega^2 \epsilon_{\vec{r}} \cdot \tilde{E}_{\vec{r}} &= -\imath \Omega \tilde{J}_{\vec{r}} e^{\imath \omega \Delta_t / 2} \\
 \Omega &= 2 \sin(\omega \Delta_t / 2) / \Delta_t
 \end{aligned}
$$

resulting in

$$
 \begin{aligned}
 \tilde{\partial}_t &\Rightarrow -\imath \Omega e^{-\imath \omega \Delta_t / 2}\\
   \hat{\partial}_t &\Rightarrow -\imath \Omega e^{ \imath \omega \Delta_t / 2}\\
 \end{aligned}
$$

Maxwell's equations are then

$$
  \begin{aligned}
  \tilde{\nabla} \times \tilde{E}_{\vec{r}} &=
         \imath \Omega e^{-\imath \omega \Delta_t / 2} \hat{B}_{\vec{r} + \frac{1}{2}}
                                                     - \hat{M}_{\vec{r} + \frac{1}{2}}  \\
  \hat{\nabla} \times \hat{H}_{\vec{r} + \frac{1}{2}} &=
        -\imath \Omega e^{ \imath \omega \Delta_t / 2} \tilde{D}_{\vec{r}}
                                                     + \tilde{J}_{\vec{r}} \\
  \tilde{\nabla} \cdot \hat{B}_{\vec{r} + \frac{1}{2}} &= 0 \\
  \hat{\nabla} \cdot \tilde{D}_{\vec{r}} &= \rho_{\vec{r}}
 \end{aligned}
$$

With $\Delta_t \to 0$, this simplifies to

$$
 \begin{aligned}
 \tilde{E}_{l, \vec{r}} &\to \tilde{E}_{\vec{r}} \\
 \tilde{H}_{l - \frac{1}{2}, \vec{r} + \frac{1}{2}} &\to \tilde{H}_{\vec{r} + \frac{1}{2}} \\
 \tilde{J}_{l, \vec{r}} &\to \tilde{J}_{\vec{r}} \\
 \tilde{M}_{l - \frac{1}{2}, \vec{r} + \frac{1}{2}} &\to \tilde{M}_{\vec{r} + \frac{1}{2}} \\
 \Omega &\to \omega \\
 \tilde{\partial}_t &\to -\imath \omega \\
   \hat{\partial}_t &\to -\imath \omega \\
 \end{aligned}
$$

and then

$$
  \begin{aligned}
  \tilde{\nabla} \times \tilde{E}_{\vec{r}} &=
         \imath \omega \hat{B}_{\vec{r} + \frac{1}{2}}
                     - \hat{M}_{\vec{r} + \frac{1}{2}}  \\
  \hat{\nabla} \times \hat{H}_{\vec{r} + \frac{1}{2}} &=
        -\imath \omega \tilde{D}_{\vec{r}}
                     + \tilde{J}_{\vec{r}} \\
 \end{aligned}
$$

$$
 \hat{\nabla} \times (\mu^{-1}_{\vec{r} + \frac{1}{2}} \cdot \tilde{\nabla} \times \tilde{E}_{\vec{r}})
    -\omega^2 \epsilon_{\vec{r}} \cdot \tilde{E}_{\vec{r}} = -\imath \omega \tilde{J}_{\vec{r}} \\
$$

# TODO FDFD?
# TODO PML


"""
from . import solvers, operators, functional, scpml, waveguide_2d, waveguide_3d
# from . import farfield, bloch TODO
