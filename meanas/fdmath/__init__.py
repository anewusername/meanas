"""

Basic discrete calculus for finite difference (fd) simulations.

TODO: short description of functional vs operator form

Discrete calculus
=================

This documentation and approach is roughly based on W.C. Chew's excellent
"Electromagnetic Theory on a Lattice" (doi:10.1063/1.355770),
which covers a superset of this material with similar notation and more detail.


Scalar derivatives and cell shifts
----------------------------------

Define the discrete forward derivative as
 $$ [\\tilde{\\partial}_x f ]_{m + \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m + 1} - f_m) $$
 where \\( f \\) is a function defined at discrete locations on the x-axis (labeled using \\( m \\)).
 The value at \\( m \\) occupies a length \\( \\Delta_{x, m} \\) along the x-axis. Note that \\( m \\)
 is an index along the x-axis, _not_ necessarily an x-coordinate, since each length
 \\( \\Delta_{x, m}, \\Delta_{x, m+1}, ...\\) is independently chosen.

If we treat `f` as a 1D array of values, with the `i`-th value `f[i]` taking up a length `dx[i]`
along the x-axis, the forward derivative is

    deriv_forward(f)[i] = (f[i + 1] - f[i]) / dx[i]


Likewise, discrete reverse derivative is
 $$ [\\hat{\\partial}_x f ]_{m - \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m} - f_{m - 1}) $$
 or

    deriv_back(f)[i] = (f[i] - f[i - 1]) / dx[i]

The derivatives' values are shifted by a half-cell relative to the original function, and
will have different cell widths if all the `dx[i]` ( \\( \\Delta_{x, m} \\) ) are not
identical:

    [figure: derivatives and cell sizes]
        dx0   dx1      dx2      dx3      cell sizes for function
       ----- ----- ----------- -----
       ______________________________
            |     |           |     |
         f0 |  f1 |     f2    |  f3 |    function
       _____|_____|___________|_____|
         |     |        |        |
         | Df0 |   Df1  |   Df2  | Df3   forward derivative (periodic boundary)
       __|_____|________|________|___

     dx'3] dx'0   dx'1     dx'2  [dx'3   cell sizes for forward derivative
       -- ----- -------- -------- ---
     dx'0] dx'1   dx'2     dx'3  [dx'0   cell sizes for reverse derivative
       ______________________________
         |     |        |        |
         | df1 |  df2   |   df3  | df0   reverse derivative (periodic boundary)
       __|_____|________|________|___

    Periodic boundaries are used here and elsewhere unless otherwise noted.

In the above figure,
 `f0 =` \\(f_0\\), `f1 =` \\(f_1\\)
 `Df0 =` \\([\\tilde{\\partial}f]_{0 + \\frac{1}{2}}\\)
 `Df1 =` \\([\\tilde{\\partial}f]_{1 + \\frac{1}{2}}\\)
 `df0 =` \\([\\hat{\\partial}f]_{0 - \\frac{1}{2}}\\)
 etc.

The fractional subscript \\( m + \\frac{1}{2} \\) is used to indicate values defined
 at shifted locations relative to the original \\( m \\), with corresponding lengths
 $$ \\Delta_{x, m + \\frac{1}{2}} = \\frac{1}{2} * (\\Delta_{x, m} + \\Delta_{x, m + 1}) $$
Just as \\( m \\) is not itself an x-coordinate, neither is \\( m + \\frac{1}{2} \\);
carefully note the positions of the various cells in the above figure vs their labels.
If the positions labeled with \\( m \\) are considered the "base" or "original" grid,
the positions labeled with \\( m + \\frac{1}{2} \\) are said to lie on a "dual" or
"derived" grid.

For the remainder of the `Discrete calculus` section, all figures will show
constant-length cells in order to focus on the vector derivatives themselves.
See the `Grid description` section below for additional information on this topic
and generalization to three dimensions.


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
     z y        Dz  /          |  /      is the set of all three
     |/_x        | Dy          | /       and is said to be "located" at (m,n,p)
                 |/            |/
        (m, n, p)|_____Dx______| (m+1, n, p)



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
         z y            <==|== :.........|.====>    located at the face centers.
         |/_x              |  /          |  /
                           | /    //     | /       Note that in a nonuniform grid, each
                           |/    // ||   |/        dimension is normalized by the cell width.
      (m-1/2, n-1/2, p-1/2)|____//_______| (m+1/2, n-1/2, p-1/2)
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
        z y                   :    ^^       |
        |/_x                  :....||.<.....|  (m, n+1, p+1/2)
                              /    ||      /
                           | v     ||   | ^
                           |/           |/
             (m, n, p+1/2) |_____>______|  (m+1, n, p+1/2)



Maxwell's Equations
===================

If we discretize both space (m,n,p) and time (l), Maxwell's equations become

 $$ \\begin{align*}
  \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}} &= -\\tilde{\\partial}_t \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}
                                                                         + \\hat{M}_{l-1, \\vec{r} + \\frac{1}{2}}  \\\\
  \\hat{\\nabla} \\times \\hat{H}_{l,\\vec{r} + \\frac{1}{2}} &= \\hat{\\partial}_t \\tilde{D}_{l, \\vec{r}}
                                                                   + \\tilde{J}_{l-\\frac{1}{2},\\vec{r}} \\\\
  \\tilde{\\nabla} \\cdot \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}} &= 0 \\\\
  \\hat{\\nabla} \\cdot \\tilde{D}_{l,\\vec{r}} &= \\rho_{l,\\vec{r}}
 \\end{align*} $$

 with

 $$ \\begin{align*}
  \\hat{B}_\\vec{r} &= \\mu_{\\vec{r} + \\frac{1}{2}} \\cdot \\hat{H}_{\\vec{r} + \\frac{1}{2}} \\\\
  \\tilde{D}_\\vec{r} &= \\epsilon_\\vec{r} \\cdot \\tilde{E}_\\vec{r}
 \\end{align*} $$

where the spatial subscripts are abbreviated as \\( \\vec{r} = (m, n, p) \\) and
\\( \\vec{r} + \\frac{1}{2} = (m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}) \\),
\\( \\tilde{E} \\) and \\( \\hat{H} \\) are the electric and magnetic fields,
\\( \\tilde{J} \\) and \\( \\hat{M} \\) are the electric and magnetic current distributions,
and \\( \\epsilon \\) and \\( \\mu \\) are the dielectric permittivity and magnetic permeability.

The above is Yee's algorithm, written in a form analogous to Maxwell's equations.
The time derivatives can be expanded to form the update equations:

    [code: Maxwell's equations]
    H[i, j, k] -= (curl_forward(E[t])[i, j, k] - M[t, i, j, k]) /      mu[i, j, k]
    E[i, j, k] += (curl_back(   H[t])[i, j, k] + J[t, i, j, k]) / epsilon[i, j, k]

Note that the E-field fore-vector and H-field back-vector are offset by a half-cell, resulting
in distinct locations for all six E- and H-field components:

    [figure: Field components]

            (m - 1/2,=> ____________Hx__________[H] <= r + 1/2 = (m + 1/2,
             n + 1/2,  /:           /:          /|                n + 1/2,
       z y   p + 1/2) / :          / :         / |                p + 1/2)
       |/_x          /  :         /  :        /  |
                    /   :       Ez__________Hy   |      Locations of the E- and
                   /    :        :   :      /|   |      H-field components for the
     (m - 1/2,    /     :        :  Ey...../.|..Hz      [E] fore-vector at r = (m,n,p)
      n - 1/2, =>/________________________/  |  /|      (the large cube's center)
      p + 1/2)   |      :        : /      |  | / |      and [H] back-vector at r + 1/2
                 |      :        :/       |  |/  |      (the top right corner)
                 |      :       [E].......|.Ex   |
                 |      :.................|......| <= (m + 1/2, n + 1/2, p + 1/2)
                 |     /                  |     /
                 |    /                   |    /
                 |   /                    |   /         This is the Yee discretization
                 |  /                     |  /          scheme ("Yee cell").
    r - 1/2 =    | /                      | /
     (m - 1/2,   |/                       |/
      n - 1/2,=> |________________________| <= (m + 1/2, n - 1/2, p - 1/2)
      p - 1/2)

Each component forms its own grid, offset from the others:

    [figure: E-fields for adjacent cells]

                  H1__________Hx0_________H0
      z y        /:                       /|
      |/_x      / :                      / |    This figure shows H back-vector locations
               /  :                     /  |    H0, H1, etc. and their associated components
             Hy1  :                   Hy0  |    H0 = (Hx0, Hy0, Hz0) etc.
             /    :                   /    |
            /    Hz1                 /     Hz0
           H2___________Hx3_________H3     |    The equivalent drawing for E would have
           |      :                 |      |    fore-vectors located at the cube's
           |      :                 |      |    center (and the centers of adjacent cubes),
           |      :                 |      |    with components on the cube's faces.
           |      H5..........Hx4...|......H4
           |     /                  |     /
          Hz2   /                  Hz2   /
           |   /                    |   /
           | Hy6                    | Hy4
           | /                      | /
           |/                       |/
           H6__________Hx7__________H7


The divergence equations can be derived by taking the divergence of the curl equations
and combining them with charge continuity,
 $$ \\hat{\\nabla} \\cdot \\tilde{J} + \\hat{\\partial}_t \\rho = 0 $$
 implying that the discrete Maxwell's equations do not produce spurious charges.


Wave equation
-------------

Taking the backward curl of the \\( \\tilde{\\nabla} \\times \\tilde{E} \\) equation and
replacing the resulting \\( \\hat{\\nabla} \\times \\hat{H} \\) term using its respective equation,
and setting \\( \\hat{M} \\) to zero, we can form the discrete wave equation:

$$
  \\begin{align*}
  \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}} &=
      -\\tilde{\\partial}_t \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}
                          + \\hat{M}_{l-1, \\vec{r} + \\frac{1}{2}}  \\\\
  \\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}} &=
    -\\tilde{\\partial}_t \\hat{H}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}  \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}}) &=
    \\hat{\\nabla} \\times (-\\tilde{\\partial}_t \\hat{H}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}})  \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}}) &=
    -\\tilde{\\partial}_t \\hat{\\nabla} \\times \\hat{H}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}  \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}}) &=
    -\\tilde{\\partial}_t \\hat{\\partial}_t \\epsilon_\\vec{r} \\tilde{E}_{l, \\vec{r}} + \\hat{\\partial}_t \\tilde{J}_{l-\\frac{1}{2},\\vec{r}} \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l, \\vec{r}})
            + \\tilde{\\partial}_t \\hat{\\partial}_t \\epsilon_\\vec{r} \\cdot \\tilde{E}_{l, \\vec{r}}
            &= \\tilde{\\partial}_t \\tilde{J}_{l - \\frac{1}{2}, \\vec{r}}
  \\end{align*}
$$



Grid description
================

As described in the section on scalar discrete derivatives above, cell widths
(`dx[i]`, `dy[j]`, `dz[k]`) along each axis can be arbitrary and independently
defined. Moreover, all field components are actually defined at "derived" or "dual"
positions, in-between the "base" grid points on one or more axes.

To get a better sense of how this works, let's start by drawing a grid with uniform
`dy` and `dz` and nonuniform `dx`. We will only draw one cell in the y and z dimensions
to make the illustration simpler; we need at least two cells in the x dimension to
demonstrate how nonuniform `dx` affects the various components.

Place the E fore-vectors at integer indices \\( r = (m, n, p) \\) and the H back-vectors
at fractional indices \\( r + \\frac{1}{2} = (m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}.
Remember that these are indices and not coordinates; they can coorespond to arbitrary
(monotonically increasing) coordinates depending on the cell widths.

Draw lines to denote the planes on which the H components and back-vectors are defined.
For simplicity, don't draw the equivalent planes for the E components and fore-vectors,
except as necessary to show their locations -- it's easiest to just connect them to their
associated H-equivalents. The result looks something like this:

    [figure: Component centers]
                                                                    p=
              [H]__________Hx___________[H]______Hx______[H]   __ +1/2
      z y     /:           /:           /:       /:      /|     |      |
      |/_x   / :          / :          / :      / :     / |     |      |
            /  :         /  :         /  :     /  :    /  |     |      |
          Hy   :       Ez...........Hy   :   Ez......Hy   |     |      |
          /:   :        :   :       /:   :    :   :  /|   |     |      |
         / :  Hz        :  Ey....../.:..Hz    :  Ey./.|..Hz    __ 0    | dz[0]
        /  :  /:        :  /      /  :  /:    :  / /  |  /|     |      |
       /_________________________/________________/   | / |     |      |
       |   :/  :        :/       |   :/  :    :/  |   |/  |     |      |
       |  Ex   :       [E].......|..Ex   :   [E]..|..Ex   |     |      |
       |       :                 |       :        |       |     |      |
       |      [H]..........Hx....|......[H].....Hx|......[H]   __ --------- (m=+1/2, p=-1/2)
       |      /                  |      /         |      /     /       /
       |     /                   |     /          |     /     /       /
      Hz    /                   Hz    /          Hz    /     /       /
       |  Hy                     |  Hy            |  Hy    __ 0     / dy[0]
       |  /                      |  /             |  /     /       /
       | /                       | /              | /     /       /
       |/                        |/               |/     /       /
      [H]__________Hx___________[H]______Hx______[H]   __ -1/2  /
                                                           =n
       |------------|------------|--------|------|
     -1/2           0          +1/2      +1    +3/2 = m

        ------------------------- ----------------
                  dx[0]                  dx[1]

      Part of a nonuniform "base grid", with labels specifying
      positions of the various field components. [E] fore-vectors
      are at the cell centers, and [H] back-vectors are at the
      vertices. H components along the near (-y) top (+z) edge
      have been omitted to make the insides of the cubes easier
      to visualize.

This figure shows where all the components are located; however, it is also useful to show
what volumes those components are responsible for representing. Consider the Ex component:
two of its nearest neighbors are E fore-vectors, labeled `[E]` in the figure.

    [figure: Ex volumes]
              <__________________________________________>
     z y     <<           /:           /    /:          >>    |
     |/_x   < <          / :          /    / :         > >    |
           <  <         /  :         /    /  :        >  >    |
          <   <        /   :        /    /   :       >   >    |
         <:   <       /    :        :   /    :      >:   >    |
        < :   <      /     :        :  /     :     > :   >    | dz[0]
       <  :   <     /      :        : /      :    >  :   >    |
      <____________/_____________________________>   :   >    |
      <   :   <    |       :        :|       :   >   :   >    |
      <  Ex   <    |       :       Ex|       :   >  Ex   >    |
      <   :   <    |       :        :|       :   >   :   >    |
      <   :   <....|.......:........:|.......:...>...:...>
      <   :  <     |      /         :|  /   /    >   :  >     /
      <   : <      |     /          :| /   /     >   : >     /
      <   :<       |    /           :|/   /      >   :>     /
      <   <        |   /            :|   /       >   >     /
      <  <         |  /              |  /        >  >     / dy[0]
      < <          | /               | /         > >     /
      <<           |/                |/          >>     /
      <____________|_________________|___________>     /

      ~------------ ----------------- -----------~
         dx'[-1]          dx'[0]          dx'[1]

     The Ex values are positioned on the x-faces of the base
     grid. They represent the Ex field in volumes shifted by
     a half-cell in the x-dimension, as shown here. Only the
     center cell is fully shown; the other two are truncated
     (shown using >< markers).

     Note that the Ex positions are the in the same positions
     as the previous figure; only the cell boundaries have moved.
     Also note that the points at which Ex is defined are not
     necessarily centered in the volumes they represent; non-
     uniform cell sizes result in off-center volumes like the
     center cell here.

    [figure: Hy volumes]
      z y    mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm    s
      |/_x  <<           m:                    m:      >>    |
           < <          m :                   m :     > >    | dz'[1]
         Hy............m...........Hy........m......Hy  >    |
         <   <        m   :                 m   :   >   >    |
        <    <       m    :                m    :  >    >    |
       <     _______m_____:_______________m_____:_>______
      mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm      >    |
      <      <     |    / :              |    / :>      >    |
      <      <     |   /  :              |   /  :>      >    | dz'[0]
      <      <     |  /   :              |  /   :>      >    |
      <      wwwwww|w/wwwwwwwwwwwwwwwwwww|w/wwwww>wwwwwww    s
      <     <      |/    w               |/    w >     >    /
      _____________|_____________________|________    >    /
      <  Hy........|...w...........Hy....|...w...>..Hy    /
      <  <         |  w                  |  w    >  >    / dy[0]
      < <          | w                   | w     > >    /
      <<           |w                    |w      >>    /
      wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

      ~------------ --------------------- -------~
         dx'[-1]            dx'[0]          dx'[1]

     The Hy values are positioned on the y-edges of the base
     grid. Again here, the 'Hy' labels represent the same points
     as in the basic grid figure above; the edges have shifted
     by a half-cell along the x- and z-axes.

     The grid lines _|:/ are edges of the area represented by
     each Hy value, and the lines drawn using <m>.w represent
     edges where a cell's faces extend beyond the drawn area
     (i.e. where the drawing is truncated in the x- or z-
     directions).


TODO: explain dxes

    [figure: 3D base and derived grids]
              _____________________________                  _____________________________
     z y     /:          /:      /:      /|          z y          /:        /:      /:
     |/_x   / :         / :     / :     / |          |/_x        / :       / :     / :
           /  :        /  :    /  :    /  |                     /  :      /  :    /  :
          /___________________________/   | dz[1]        ________________________/____
         /    :      /    :  /    :  /|   |                   /:   :    /    :  /:   : dz[1]
        /:    :     /     : /     : / |   |                  / :   :   /     : / :   :
       / :    :..../......:/......:/..|...|                 / .:...:../......:/..:...:.....
      /___________/_______/_______/   |  /|          ______/_________/_______/___:   :
      |  :  / :   |       |       |   | / |                |   :   : |       |   :   :
      |  : /  :   |       |       |   |/  |                |   :   : |       |   :   :
      |  :/   :   |       |       |   |   | dz[0]          |   :   : |       |   :   : dz[0]
      |  /    :   |       |       |  /|   |                |   :   : |       |   :   :
      | /:    :...|.......|.......|./ |...|                | ..:...:.|.......|...:...:.....
      |/ :   /    |      /|      /|/  |  /                 |   :  /  |      /|   :  /
      |___________|_______|_______|   | /  dy[1]     ______|_________|_______|___: / dy[1]
      |  : /      |    /  |    /  |   |/                   |   :/    |    /  |   :/
      |  :/.......|.../...|.../...|...|                  ..|...:.....|.../...|...:...
      |  /        |  /    |  /    |  /                     |  /      |  /    |  / dy[0]
      | /         | /     | /     | /  dy[0]               | /       | /     | /
      |/          |/      |/      |/                       |/        |/      |/
      |___________|_______|_______|                  ______|_________|_______|___
          dx[0]     dx[1]   dx[2]                            dx'[0]   dx'[1]  dx'[2]

                Base grid                         Shifted one half-cell right (e.g. for 1D
                                                  forward x derivative of all components).
       z y         : /        : /     :dz'[1]        Some lines are omitted for clarity.
       |/_x        :/         :/      :/
            .......:..........:.......:...
               |  /:      |  /:   |  /:
               | / :      | / :   | / :
               |/  :      |/  :   |/  :dz'[0]
        ______________________________
              /|   :/    /|   :/ /|   :/dy'[1]
             /.|...:..../.|...:./.|...:....
               |  /:      |  /:   |  /:
               | / :      | / :   | /dy'[0]
               |/  :      |/  :   |/  :
        _______________________________
              /|         /|      /|
             / |        / |     / |
               |          |       |
                  dx'[0]    dx'[1]  dx'[2]

      All three dimensions shifted by one half-
      cell. This is quite hard to visualize
      (and probably not entirely to scale); see
      later figures for a better representation.


"""

from .types import fdfield_t, vfdfield_t, dx_lists_t, fdfield_updater_t
from .vectorization import vec, unvec
from . import operators, functional, types, vectorization

