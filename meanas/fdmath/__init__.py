"""

Basic discrete calculus for finite difference (fd) simulations.

Discrete calculus
=================

This documentation and approach is roughly based on W.C. Chew's excellent
"Electromagnetic Theory on a Lattice" (doi:10.1063/1.355770),
which covers a superset of this material with similar notation and more detail.


Derivatives
-----------

Define the discrete forward derivative as
 $$ [\\tilde{\\partial}_x f ]_{m + \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m + 1} - f_m) $$
 or

    Dx_forward(f)[i] = (f[i + 1] - f[i]) / dx[i]

Likewise, discrete reverse derivative is
 $$ [\\hat{\\partial}_x f ]_{m - \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m} - f_{m - 1}) $$

 or

    Dx_back(f)[i] = (f[i] - f[i - 1]) / dx[i]

The derivatives' arrays are shifted by a half-cell relative to the original function:

    [figure: derivatives]
     _________________________
     |     |     |     |     |
     |  f0 |  f1 |  f2 |  f3 |      function
     |_____|_____|_____|_____|
        |     |     |     |
        | Df0 | Df1 | Df2 | Df3     forward derivative (periodic boundary)
     ___|_____|_____|_____|____
        |     |     |     |
        | Df1 | Df2 | Df3 | Df0     reverse derivative (periodic boundary)
     ___|_____|_____|_____|____

Periodic boundaries are used unless otherwise noted.


Gradients and fore-vectors
--------------------------

Expanding to three dimensions, we can define two gradients
  $$ [\\tilde{\\nabla} f]_{m,n,p} = \\vec{x} [\\tilde{\\partial}_x f]_{m + \\frac{1}{2},n,p} +
                                    \\vec{y} [\\tilde{\\partial}_y f]_{m,n + \\frac{1}{2},p} +
                                    \\vec{z} [\\tilde{\\partial}_z f]_{m,n,p + \\frac{1}{2}}  $$
  $$ [\\hat{\\nabla} f]_{m,n,p} = \\vec{x} [\\hat{\\partial}_x f]_{m + \\frac{1}{2},n,p} +
                                  \\vec{y} [\\hat{\\partial}_y f]_{m,n + \\frac{1}{2},p} +
                                  \\vec{z} [\\hat{\\partial}_z f]_{m,n,p + \\frac{1}{2}}  $$

 or

    [code: gradients]
    grad_forward(f)[i,j,k] = [Dx_forward(f)[i, j, k],
                              Dy_forward(f)[i, j, k],
                              Dz_forward(f)[i, j, k]]
                           = [(f[i + 1, j, k] - f[i, j, k]) / dx[i],
                              (f[i, j + 1, k] - f[i, j, k]) / dy[i],
                              (f[i, j, k + 1] - f[i, j, k]) / dz[i]]

    grad_back(f)[i,j,k] = [Dx_back(f)[i, j, k],
                           Dy_back(f)[i, j, k],
                           Dz_back(f)[i, j, k]]
                        = [(f[i, j, k] - f[i - 1, j, k]) / dx[i],
                           (f[i, j, k] - f[i, j - 1, k]) / dy[i],
                           (f[i, j, k] - f[i, j, k - 1]) / dz[i]]

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


    [figure: gradient / fore-vector]
       (m, n+1, p+1) ______________ (m+1, n+1, p+1)
                    /:            /|
                   / :           / |
                  /  :          /  |
      (m, n, p+1)/_____________/   |     The forward derivatives are defined
                 |   :         |   |     at the Dx, Dy, Dz points,
                 |   :.........|...|     but the forward-gradient fore-vector
                Dz   /         |   /     is the set of all three
                 |  Dy         |  /      and is said to be "located" at (m,n,p)
                 | /           | /
        (m, n, p)|/_____Dx_____|/ (m+1, n, p)



Divergences
-----------

There are also two divergences,

  $$ d_{n,m,p} = [\\tilde{\\nabla} \\cdot \\hat{g}]_{n,m,p}
               = [\\tilde{\\partial}_x g^x]_{m,n,p} +
                 [\\tilde{\\partial}_y g^y]_{m,n,p} +
                 [\\tilde{\\partial}_z g^z]_{m,n,p}   $$

  $$ d_{n,m,p} = [\\hat{\\nabla} \\cdot \\tilde{g}]_{n,m,p}
               = [\\hat{\\partial}_x g^x]_{m,n,p} +
                 [\\hat{\\partial}_y g^y]_{m,n,p} +
                 [\\hat{\\partial}_z g^z]_{m,n,p}  $$

 or

    [code: divergences]
    div_forward(g)[i,j,k] = Dx_forward(gx)[i, j, k] +
                            Dy_forward(gy)[i, j, k] +
                            Dz_forward(gz)[i, j, k]
                          = (gx[i + 1, j, k] - gx[i, j, k]) / dx[i] +
                            (gy[i, j + 1, k] - gy[i, j, k]) / dy[i] +
                            (gz[i, j, k + 1] - gz[i, j, k]) / dz[i]

    div_back(g)[i,j,k] = Dx_back(gx)[i, j, k] +
                         Dy_back(gy)[i, j, k] +
                         Dz_back(gz)[i, j, k]
                       = (gx[i, j, k] - gx[i - 1, j, k]) / dx[i] +
                         (gy[i, j, k] - gy[i, j - 1, k]) / dy[i] +
                         (gz[i, j, k] - gz[i, j, k - 1]) / dz[i]

where `g = [gx, gy, gz]` is a fore- or back-vector field.

Since we applied the forward divergence to the back-vector (and vice-versa), the resulting scalar value
is defined at the back-vector's (fore-vectors) location \\( (m,n,p) \\) and not at the locations of its components
\\( (m \\pm \\frac{1}{2},n,p) \\) etc.

    [figure: divergence]
                                    ^^
         (m-1/2, n+1/2, p+1/2) _____||_______ (m+1/2, n+1/2, p+1/2)
                              /:    ||  ,,  /|
                             / :    || //  / |      The divergence at (m, n, p) (the center
                            /  :      //  /  |      of this cube) of a fore-vector field
      (m-1/2, n-1/2, p+1/2)/_____________/   |      is the sum of the outward-pointing
                           |   :         |   |      fore-vector components, which are
                        <==|== :.........|.====>    located at the face centers.
                           |   /         |   /
                           |  /   //     |  /       Note that in a nonuniform grid, each
                           | /   // ||   | /        dimension is normalized by the cell width.
      (m-1/2, n-1/2, p-1/2)|/___//_______|/ (m+1/2, n-1/2, p-1/2)
                               ''   ||
                                    VV


Curls
-----

The two curls are then

  $$ \\begin{align*}
     \\hat{h}_{m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}} &= \\\\
     [\\tilde{\\nabla} \\times \\tilde{g}]_{m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}} &=
        \\vec{x} (\\tilde{\\partial}_y g^z_{m,n,p + \\frac{1}{2}} - \\tilde{\\partial}_z g^y_{m,n + \\frac{1}{2},p}) \\\\
     &+ \\vec{y} (\\tilde{\\partial}_z g^x_{m + \\frac{1}{2},n,p} - \\tilde{\\partial}_x g^z_{m,n,p + \\frac{1}{2}}) \\\\
     &+ \\vec{z} (\\tilde{\\partial}_x g^y_{m,n + \\frac{1}{2},p} - \\tilde{\\partial}_y g^z_{m + \\frac{1}{2},n,p})
     \\end{align*} $$

 and

  $$ \\tilde{h}_{m - \\frac{1}{2}, n - \\frac{1}{2}, p - \\frac{1}{2}} =
     [\\hat{\\nabla} \\times \\hat{g}]_{m - \\frac{1}{2}, n - \\frac{1}{2}, p - \\frac{1}{2}} $$

  where \\( \\hat{g} \\) and \\( \\tilde{g} \\) are located at \\((m,n,p)\\)
  with components at  \\( (m \\pm \\frac{1}{2},n,p) \\) etc.,
  while \\( \\hat{h} \\) and \\( \\tilde{h} \\) are located at \\((m \\pm \\frac{1}{2}, n \\pm \\frac{1}{2}, p \\pm \\frac{1}{2})\\)
  with components at \\((m, n \\pm \\frac{1}{2}, p \\pm \\frac{1}{2})\\) etc.


    [code: curls]
    curl_forward(g)[i,j,k] = [Dy_forward(gz)[i, j, k] - Dz_forward(gy)[i, j, k],
                              Dz_forward(gx)[i, j, k] - Dx_forward(gz)[i, j, k],
                              Dx_forward(gy)[i, j, k] - Dy_forward(gx)[i, j, k]]

    curl_back(g)[i,j,k] = [Dy_back(gz)[i, j, k] - Dz_back(gy)[i, j, k],
                           Dz_back(gx)[i, j, k] - Dx_back(gz)[i, j, k],
                           Dx_back(gy)[i, j, k] - Dy_back(gx)[i, j, k]]


For example, consider the forward curl, at (m, n, p), of a back-vector field `g`, defined
 on a grid containing (m + 1/2, n + 1/2, p + 1/2).
 The curl will be a fore-vector, so its z-component will be defined at (m, n, p + 1/2).
 Take the nearest x- and y-components of `g` in the xy plane where the curl's z-component
 is located; these are

    [curl components]
    (m,       n + 1/2, p + 1/2) : x-component of back-vector at (m + 1/2, n + 1/2, p + 1/2)
    (m + 1,   n + 1/2, p + 1/2) : x-component of back-vector at (m + 3/2, n + 1/2, p + 1/2)
    (m + 1/2, n      , p + 1/2) : y-component of back-vector at (m + 1/2, n + 1/2, p + 1/2)
    (m + 1/2, n + 1  , p + 1/2) : y-component of back-vector at (m + 1/2, n + 3/2, p + 1/2)

 These four xy-components can be used to form a loop around the curl's z-component; its magnitude and sign
 is set by their loop-oriented sum (i.e. two have their signs flipped to complete the loop).

    [figure: z-component of curl]
                               :             |
                               :    ^^       |
                               :....||.<.....|  (m, n+1, p+1/2)
                               /    ||       /
                           |  v     ||   |  ^
                           | /           | /
             (m, n, p+1/2) |/_____>______|/ (m+1, n, p+1/2)



Maxwell's Equations
===================

If we discretize both space (m,n,p) and time (l), Maxwell's equations become

 $$ \\begin{align*}
  \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}} &=& -&\\tilde{\\partial}_t \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}
                                                                         &+& \\hat{M}_{l-1, \\vec{r} + \\frac{1}{2}}  \\\\
  \\hat{\\nabla} \\times \\hat{H}_{l,\\vec{r}} &=& &\\hat{\\partial}_t \\tilde{D}_{l, \\vec{r}}
                                                                   &+& \\tilde{J}_{l-\\frac{1}{2},\\vec{r}} \\\\
  \\tilde{\\nabla} \\cdot \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}} &= 0 \\\\
  \\hat{\\nabla} \\cdot \\tilde{D}_{l,\\vec{r}} &= \\rho_{l,\\vec{r}}
 \\end{align*} $$

 with

 $$ \\begin{align*}
  \\hat{B}_\\vec{r} &= \\mu_{\\vec{r} + \\frac{1}{2}} \\cdot \\hat{H}_{\\vec{r} + \\frac{1}{2}} \\\\
  \\tilde{D}_\\vec{r} &= \\epsilon_\\vec{r} \\cdot \\tilde{E}_\\vec{r}
 \\end{align*} $$

where the spatial subscripts are abbreviated as \\( \\vec{r} = (m, n, p) \\) and
\\( \\vec{r} + \\frac{1}{2} = (m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}) \\).
This is Yee's algorithm, written in a form analogous to Maxwell's equations.

The divergence equations can be derived by taking the divergence of the curl equations
and combining them with charge continuity,
 $$ \\hat{\\nabla} \\cdot \\tilde{J} + \\hat{\\partial}_t \\rho = 0 $$
 implying that the discrete Maxwell's equations do not produce spurious charges.

TODO: Maxwell's equations explanation
TODO: Maxwell's equations plaintext

Wave equation
-------------

$$
 \\hat{\\nabla} \\times \\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l, \\vec{r}}
            + \\tilde{\\partial}_t \\hat{\\partial}_t \\epsilon_\\vec{r} \\cdot \\tilde{E}_{l, \\vec{r}}
            = \\tilde{\\partial}_t \\tilde{J}_{l - \\frac{1}{2}, \\vec{r}} $$

TODO: wave equation explanation
TODO: wave equation plaintext


Grid description
================
TODO: explain dxes

"""

from .types import fdfield_t, vfdfield_t, dx_lists_t, fdfield_updater_t
from .vectorization import vec, unvec
from . import operators, functional, types, vectorization

