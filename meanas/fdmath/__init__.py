"""
Basic discrete calculus for finite difference (fd) simulations.

This documentation and approach is roughly based on W.C. Chew's excellent
"Electromagnetic Theory on a Lattice" (doi:10.1063/1.355770),
which covers a superset of this material with similar notation and more detail.


Define the discrete forward derivative as

    Dx_forward(f)[i] = (f[i + 1] - f[i]) / dx[i]

or
    $$ [\\tilde{\\partial}_x f ]_{m + \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m + 1} - f_m) $$

Likewise, discrete reverse derivative is

    Dx_back(f)[i] = (f[i] - f[i - 1]) / dx[i]

or
    $$ [\\hat{\\partial}_x f ]_{m - \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m} - f_{m - 1}) $$


The derivatives are shifted by a half-cell relative to the original function:

     _________________________
     |     |     |     |     |
     |  f0 |  f1 |  f2 |  f3 |
     |_____|_____|_____|_____|
        |     |     |     |
        | Df0 | Df1 | Df2 | Df3
     ___|_____|_____|_____|____

Periodic boundaries are used unless otherwise noted.


Expanding to three dimensions, we can define two gradients
  $$ [\\tilde{\\nabla} f]_{n,m,p} = \\vec{x} [\\tilde{\\partial}_x f]_{m + \\frac{1}{2},n,p} +
                                    \\vec{y} [\\tilde{\\partial}_y f]_{m,n + \\frac{1}{2},p} +
                                    \\vec{z} [\\tilde{\\partial}_z f]_{m,n,p + \\frac{1}{2}}  $$
  $$ [\\hat{\\nabla} f]_{m,n,p} = \\vec{x} [\\hat{\\partial}_x f]_{m + \\frac{1}{2},n,p} +
                                  \\vec{y} [\\hat{\\partial}_y f]_{m,n + \\frac{1}{2},p} +
                                  \\vec{z} [\\hat{\\partial}_z f]_{m,n,p + \\frac{1}{2}}  $$

The three derivatives in the gradient cause shifts in different
directions, so the x/y/z components of the resulting "vector" are defined
at different points: the x-component is shifted in the x-direction,
y in y, and z in z.

We call the resulting object a "fore-vector" or "back-vector", depending
on the direction of the shift. We write it as
  $$ \\tilde{g}_{m,n,p} = \\vec{x} g^x_{m + \\frac{1}{2},n,p} +
                          \\vec{y} g^y_{m,n + \\frac{1}{2},p} +
                          \\vec{z} g^z_{m,n,p + \\frac{1}{2}} $$
  $$ \\hat{g}_{m,n,p} = \\vec{x} g^x_{m - \\frac{1}{2},n,p} +
                        \\vec{y} g^y_{m,n - \\frac{1}{2},p} +
                        \\vec{z} g^z_{m,n,p - \\frac{1}{2}} $$


There are also two divergences,

  $$ d_{n,m,p} = [\\tilde{\\nabla} \\cdot \\hat{g}]_{n,m,p}
               = [\\tilde{\\partial}_x g^x]_{m,n,p} +
                 [\\tilde{\\partial}_y g^y]_{m,n,p} +
                 [\\tilde{\\partial}_z g^z]_{m,n,p}   $$

  $$ d_{n,m,p} = [\\hat{\\nabla} \\cdot \\tilde{g}]_{n,m,p}
               = [\\hat{\\partial}_x g^x]_{m,n,p} +
                 [\\hat{\\partial}_y g^y]_{m,n,p} +
                 [\\hat{\\partial}_z g^z]_{m,n,p}  $$

Since we applied the forward divergence to the back-vector (and vice-versa), the resulting scalar value
is defined at the back-vector's (fore-vectors) location \\( (m,n,p) \\) and not at the locations of its components
\\( (m \\pm \\frac{1}{2},n,p) \\) etc.


The two curls are then
  $$ \\begin{align}
     \\hat{h}_{m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}} &= \\\\
     [\\tilde{\\nabla} \\times \\tilde{g}]_{m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}} &=
        \\vec{x} (\\tilde{\\partial}_y g^z_{m,n,p + \\frac{1}{2}} - \\tilde{\\partial}_z g^y_{m,n + \\frac{1}{2},p}) \\\\
     &+ \\vec{y} (\\tilde{\\partial}_z g^x_{m + \\frac{1}{2},n,p} - \\tilde{\\partial}_x g^z_{m,n,p + \\frac{1}{2}}) \\\\
     &+ \\vec{z} (\\tilde{\\partial}_x g^y_{m,n + \\frac{1}{2},p} - \\tilde{\\partial}_x g^z_{m + \\frac{1}{2},n,p})
     \\end{align}$$
and
  $$ \\tilde{h}_{m - \\frac{1}{2}, n - \\frac{1}{2}, p - \\frac{1}{2}} =
     [\\hat{\\nabla} \\times \\hat{g}]_{m - \\frac{1}{2}, n - \\frac{1}{2}, p - \\frac{1}{2}} $$

  where \\( \\hat{g} \\) and \\( \\tilde{g} \\) are located at \\((m,n,p)\\)
  with components at  \\( (m \\pm \\frac{1}{2},n,p) \\) etc.,
  while \\( \\hat{h} \\) and \\( \\tilde{h} \\) are located at \\((m \\pm \\frac{1}{2}, n \\pm \\frac{1}{2}, p \\pm \\frac{1}{2})\\)
  with components at \\((m, n \\pm \\frac{1}{2}, p \\pm \\frac{1}{2})\\) etc.

TODO: Explain fdfield_t vs vfdfield_t  / operators vs functional
TODO: explain dxes

"""

from .types import fdfield_t, vfdfield_t, dx_lists_t, fdfield_updater_t
from .vectorization import vec, unvec
from . import operators, functional, types, vectorization

