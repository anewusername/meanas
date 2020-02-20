"""
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
 \\begin{aligned}
 \\tilde{E}_{l, \\vec{r}} &= \\tilde{E}_{\\vec{r}} e^{-\\imath \\omega l \\Delta_t} \\\\
 \\tilde{J}_{l, \\vec{r}} &= \\tilde{J}_{\\vec{r}} e^{-\\imath \\omega (l - \\frac{1}{2}) \\Delta_t} \\\\
 \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{\\vec{r}})
    -\\Omega^2 \\epsilon_{\\vec{r}} \\cdot \\tilde{E}_{\\vec{r}} &= \\imath \\Omega \\tilde{J}_{\\vec{r}} \\\\
 \\Omega &= 2 \\sin(\\omega \\Delta_t / 2) / \\Delta_t
 \\end{aligned}
$$


# TODO FDFD?
# TODO PML


"""
from . import solvers, operators, functional, scpml, waveguide_2d, waveguide_3d
# from . import farfield, bloch TODO
