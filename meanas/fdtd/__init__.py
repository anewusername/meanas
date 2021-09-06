"""
Utilities for running finite-difference time-domain (FDTD) simulations

See the discussion of `Maxwell's Equations` in `meanas.fdmath` for basic
mathematical background.


Timestep
========

From the discussion of "Plane waves and the Dispersion relation" in `meanas.fdmath`,
we have

$$ c^2 \\Delta_t^2 = \\frac{\\Delta_t^2}{\\mu \\epsilon} < 1/(\\frac{1}{\\Delta_x^2} + \\frac{1}{\\Delta_y^2} + \\frac{1}{\\Delta_z^2}) $$

or, if $\\Delta_x = \\Delta_y = \\Delta_z$, then $c \\Delta_t < \\frac{\\Delta_x}{\\sqrt{3}}$.

Based on this, we can set

    dt = sqrt(mu.min() * epsilon.min()) / sqrt(1/dx_min**2 + 1/dy_min**2 + 1/dz_min**2)

The `dx_min`, `dy_min`, `dz_min` should be the minimum value across both the base and derived grids.


Poynting Vector and Energy Conservation
=======================================

Let

$$ \\begin{aligned}
  \\tilde{S}_{l, l', \\vec{r}} &=& &\\tilde{E}_{l, \\vec{r}} \\otimes \\hat{H}_{l', \\vec{r} + \\frac{1}{2}}  \\\\
  &=&  &\\vec{x} (\\tilde{E}^y_{l,m+1,n,p} \\hat{H}^z_{l',\\vec{r} + \\frac{1}{2}} - \\tilde{E}^z_{l,m+1,n,p} \\hat{H}^y_{l', \\vec{r} + \\frac{1}{2}}) \\\\
  & &+ &\\vec{y} (\\tilde{E}^z_{l,m,n+1,p} \\hat{H}^x_{l',\\vec{r} + \\frac{1}{2}} - \\tilde{E}^x_{l,m,n+1,p} \\hat{H}^z_{l', \\vec{r} + \\frac{1}{2}}) \\\\
  & &+ &\\vec{z} (\\tilde{E}^x_{l,m,n,p+1} \\hat{H}^y_{l',\\vec{r} + \\frac{1}{2}} - \\tilde{E}^y_{l,m,n,p+1} \\hat{H}^z_{l', \\vec{r} + \\frac{1}{2}})
   \\end{aligned}
$$

where $\\vec{r} = (m, n, p)$ and $\\otimes$ is a modified cross product
in which the $\\tilde{E}$ terms are shifted as indicated.

By taking the divergence and rearranging terms, we can show that

$$
  \\begin{aligned}
  \\hat{\\nabla} \\cdot \\tilde{S}_{l, l', \\vec{r}}
   &= \\hat{\\nabla} \\cdot (\\tilde{E}_{l, \\vec{r}} \\otimes \\hat{H}_{l', \\vec{r} + \\frac{1}{2}})  \\\\
   &= \\hat{H}_{l', \\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l, \\vec{r}} -
      \\tilde{E}_{l, \\vec{r}} \\cdot \\hat{\\nabla} \\times \\hat{H}_{l', \\vec{r} + \\frac{1}{2}} \\\\
   &= \\hat{H}_{l', \\vec{r} + \\frac{1}{2}} \\cdot
            (-\\tilde{\\partial}_t \\mu_{\\vec{r} + \\frac{1}{2}} \\hat{H}_{l - \\frac{1}{2}, \\vec{r} + \\frac{1}{2}} -
                \\hat{M}_{l-1, \\vec{r} + \\frac{1}{2}}) -
      \\tilde{E}_{l, \\vec{r}} \\cdot (\\hat{\\partial}_t \\tilde{\\epsilon}_{\\vec{r}} \\tilde{E}_{l'+\\frac{1}{2}, \\vec{r}} +
                \\tilde{J}_{l', \\vec{r}}) \\\\
   &= \\hat{H}_{l'} \\cdot (-\\mu / \\Delta_t)(\\hat{H}_{l + \\frac{1}{2}} - \\hat{H}_{l - \\frac{1}{2}}) -
      \\tilde{E}_l \\cdot (\\epsilon / \\Delta_t )(\\tilde{E}_{l'+\\frac{1}{2}} - \\tilde{E}_{l'-\\frac{1}{2}})
      - \\hat{H}_{l'} \\cdot \\hat{M}_{l-1} - \\tilde{E}_l \\cdot \\tilde{J}_{l'} \\\\
  \\end{aligned}
$$

where in the last line the spatial subscripts have been dropped to emphasize
the time subscripts $l, l'$, i.e.

$$
  \\begin{aligned}
  \\tilde{E}_l &= \\tilde{E}_{l, \\vec{r}} \\\\
  \\hat{H}_l &= \\tilde{H}_{l, \\vec{r} + \\frac{1}{2}}  \\\\
  \\tilde{\\epsilon} &= \\tilde{\\epsilon}_{\\vec{r}}  \\\\
  \\end{aligned}
$$

etc.
For $l' = l + \\frac{1}{2}$ we get

$$
  \\begin{aligned}
  \\hat{\\nabla} \\cdot \\tilde{S}_{l, l + \\frac{1}{2}}
   &= \\hat{H}_{l + \\frac{1}{2}} \\cdot
            (-\\mu / \\Delta_t)(\\hat{H}_{l + \\frac{1}{2}} - \\hat{H}_{l - \\frac{1}{2}}) -
      \\tilde{E}_l \\cdot (\\epsilon / \\Delta_t)(\\tilde{E}_{l+1} - \\tilde{E}_l)
      - \\hat{H}_{l'} \\cdot \\hat{M}_l - \\tilde{E}_l \\cdot \\tilde{J}_{l + \\frac{1}{2}} \\\\
   &= (-\\mu / \\Delta_t)(\\hat{H}^2_{l + \\frac{1}{2}} - \\hat{H}_{l + \\frac{1}{2}} \\cdot \\hat{H}_{l - \\frac{1}{2}}) -
      (\\epsilon / \\Delta_t)(\\tilde{E}_{l+1} \\cdot \\tilde{E}_l - \\tilde{E}^2_l)
      - \\hat{H}_{l'} \\cdot \\hat{M}_l - \\tilde{E}_l \\cdot \\tilde{J}_{l + \\frac{1}{2}} \\\\
   &= -(\\mu \\hat{H}^2_{l + \\frac{1}{2}}
       +\\epsilon \\tilde{E}_{l+1} \\cdot \\tilde{E}_l) / \\Delta_t \\ \\
      +(\\mu \\hat{H}_{l + \\frac{1}{2}} \\cdot \\hat{H}_{l - \\frac{1}{2}}
       +\\epsilon \\tilde{E}^2_l) / \\Delta_t \\ \\
      - \\hat{H}_{l+\\frac{1}{2}} \\cdot \\hat{M}_l \\ \\
      - \\tilde{E}_l \\cdot \\tilde{J}_{l+\\frac{1}{2}} \\\\
  \\end{aligned}
$$

and for $l' = l - \\frac{1}{2}$,

$$
  \\begin{aligned}
  \\hat{\\nabla} \\cdot \\tilde{S}_{l, l - \\frac{1}{2}}
   &=  (\\mu \\hat{H}^2_{l - \\frac{1}{2}}
       +\\epsilon \\tilde{E}_{l-1} \\cdot \\tilde{E}_l) / \\Delta_t \\ \\
      -(\\mu \\hat{H}_{l + \\frac{1}{2}} \\cdot \\hat{H}_{l - \\frac{1}{2}}
       +\\epsilon \\tilde{E}^2_l) / \\Delta_t \\ \\
      - \\hat{H}_{l-\\frac{1}{2}} \\cdot \\hat{M}_l \\ \\
      - \\tilde{E}_l \\cdot \\tilde{J}_{l-\\frac{1}{2}} \\\\
  \\end{aligned}
$$

These two results form the discrete time-domain analogue to Poynting's theorem.
They hint at the expressions for the energy, which can be calculated at the same
time-index as either the E or H field:

$$
 \\begin{aligned}
 U_l &= \\epsilon \\tilde{E}^2_l + \\mu \\hat{H}_{l + \\frac{1}{2}} \\cdot \\hat{H}_{l - \\frac{1}{2}} \\\\
 U_{l + \\frac{1}{2}} &= \\epsilon \\tilde{E}_l \\cdot \\tilde{E}_{l + 1} + \\mu \\hat{H}^2_{l + \\frac{1}{2}} \\\\
 \\end{aligned}
$$

Rewriting the Poynting theorem in terms of the energy expressions,

$$
  \\begin{aligned}
  (U_{l+\\frac{1}{2}} - U_l) / \\Delta_t
   &= -\\hat{\\nabla} \\cdot \\tilde{S}_{l, l + \\frac{1}{2}} \\ \\
      - \\hat{H}_{l+\\frac{1}{2}} \\cdot \\hat{M}_l \\ \\
      - \\tilde{E}_l \\cdot \\tilde{J}_{l+\\frac{1}{2}} \\\\
  (U_l - U_{l-\\frac{1}{2}}) / \\Delta_t
   &= -\\hat{\\nabla} \\cdot \\tilde{S}_{l, l - \\frac{1}{2}} \\ \\
      - \\hat{H}_{l-\\frac{1}{2}} \\cdot \\hat{M}_l \\ \\
      - \\tilde{E}_l \\cdot \\tilde{J}_{l-\\frac{1}{2}} \\\\
 \\end{aligned}
$$

This result is exact and should practically hold to within numerical precision. No time-
or spatial-averaging is necessary.

Note that each value of $J$ contributes to the energy twice (i.e. once per field update)
despite only causing the value of $E$ to change once (same for $M$ and $H$).


Sources
=============

It is often useful to excite the simulation with an arbitrary broadband pulse and then
extract the frequency-domain response by performing an on-the-fly Fourier transform
of the time-domain fields.

The Ricker wavelet (normalized second derivative of a Gaussian) is commonly used for the pulse
shape. It can be written

$$ f_r(t) = (1 - \\frac{1}{2} (\\omega (t - \\tau))^2) e^{-(\\frac{\\omega (t - \\tau)}{2})^2} $$

with $\\tau > \\frac{2 * \\pi}{\\omega}$ as a minimum delay to avoid a discontinuity at
t=0 (assuming the source is off for t<0 this gives $\\sim 10^{-3}$ error at t=0).



Boundary conditions
===================
# TODO notes about boundaries / PMLs
"""

from .base import maxwell_e, maxwell_h
from .pml import cpml_params, updates_with_cpml
from .energy import (poynting, poynting_divergence, energy_hstep, energy_estep,
                     delta_energy_h2e, delta_energy_j)
from .boundaries import conducting_boundary
