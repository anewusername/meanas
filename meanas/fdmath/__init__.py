"""

Basic discrete calculus for finite difference (fd) simulations.


Fields, Functions, and Operators
================================

Discrete fields are stored in one of two forms:

- The `fdfield_t` form is a multidimensional `numpy.ndarray`
    + For a scalar field, this is just `U[m, n, p]`, where `m`, `n`, and `p` are
      discrete indices referring to positions on the x, y, and z axes respectively.
    + For a vector field, the first index specifies which vector component is accessed:
      `E[:, m, n, p] = [Ex[m, n, p], Ey[m, n, p], Ez[m, n, p]]`.
- The `vfdfield_t` form is simply a vectorzied (i.e. 1D) version of the `field_t`,
    as obtained by `meanas.fdmath.vectorization.vec` (effectively just `numpy.ravel`)

Operators which act on fields also come in two forms:
    + Python functions, created by the functions in `meanas.fdmath.functional`.
        The generated functions act on fields in the `fdfield_t` form.
    + Linear operators, usually 2D sparse matrices using `scipy.sparse`, created
        by `meanas.fdmath.operators`. These operators act on vectorized fields in the
        `vfdfield_t` form.

The operations performed should be equivalent: `functional.op(*args)(E)` should be
equivalent to `unvec(operators.op(*args) @ vec(E), E.shape[1:])`.

Generally speaking the `field_t` form is easier to work with, but can be harder or less
efficient to compose (e.g. it is easy to generate a single matrix by multiplying a
series of other matrices).


Discrete calculus
=================

This documentation and approach is roughly based on W.C. Chew's excellent
"Electromagnetic Theory on a Lattice" (doi:10.1063/1.355770),
which covers a superset of this material with similar notation and more detail.


Scalar derivatives and cell shifts
----------------------------------

Define the discrete forward derivative as
 $$ [\\tilde{\\partial}_x f]_{m + \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m + 1} - f_m) $$
 where $f$ is a function defined at discrete locations on the x-axis (labeled using $m$).
 The value at $m$ occupies a length $\\Delta_{x, m}$ along the x-axis. Note that $m$
 is an index along the x-axis, _not_ necessarily an x-coordinate, since each length
 $\\Delta_{x, m}, \\Delta_{x, m+1}, ...$ is independently chosen.

If we treat `f` as a 1D array of values, with the `i`-th value `f[i]` taking up a length `dx[i]`
along the x-axis, the forward derivative is

    deriv_forward(f)[i] = (f[i + 1] - f[i]) / dx[i]


Likewise, discrete reverse derivative is
 $$ [\\hat{\\partial}_x f ]_{m - \\frac{1}{2}} = \\frac{1}{\\Delta_{x, m}} (f_{m} - f_{m - 1}) $$
 or

    deriv_back(f)[i] = (f[i] - f[i - 1]) / dx[i]

The derivatives' values are shifted by a half-cell relative to the original function, and
will have different cell widths if all the `dx[i]` ( $\\Delta_{x, m}$ ) are not
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
 `f0 =` $f_0$, `f1 =` $f_1$
 `Df0 =` $[\\tilde{\\partial}f]_{0 + \\frac{1}{2}}$
 `Df1 =` $[\\tilde{\\partial}f]_{1 + \\frac{1}{2}}$
 `df0 =` $[\\hat{\\partial}f]_{0 - \\frac{1}{2}}$
 etc.

The fractional subscript $m + \\frac{1}{2}$ is used to indicate values defined
 at shifted locations relative to the original $m$, with corresponding lengths
 $$ \\Delta_{x, m + \\frac{1}{2}} = \\frac{1}{2} * (\\Delta_{x, m} + \\Delta_{x, m + 1}) $$

Just as $m$ is not itself an x-coordinate, neither is $m + \\frac{1}{2}$;
carefully note the positions of the various cells in the above figure vs their labels.
If the positions labeled with $m$ are considered the "base" or "original" grid,
the positions labeled with $m + \\frac{1}{2}$ are said to lie on a "dual" or
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
is defined at the back-vector's (fore-vector's) location $(m,n,p)$ and not at the locations of its components
$(m \\pm \\frac{1}{2},n,p)$ etc.

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

  $$ \\begin{aligned}
     \\hat{h}_{m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}} &= \\\\
     [\\tilde{\\nabla} \\times \\tilde{g}]_{m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2}} &=
        \\vec{x} (\\tilde{\\partial}_y g^z_{m,n,p + \\frac{1}{2}} - \\tilde{\\partial}_z g^y_{m,n + \\frac{1}{2},p}) \\\\
     &+ \\vec{y} (\\tilde{\\partial}_z g^x_{m + \\frac{1}{2},n,p} - \\tilde{\\partial}_x g^z_{m,n,p + \\frac{1}{2}}) \\\\
     &+ \\vec{z} (\\tilde{\\partial}_x g^y_{m,n + \\frac{1}{2},p} - \\tilde{\\partial}_y g^z_{m + \\frac{1}{2},n,p})
     \\end{aligned} $$

 and

  $$ \\tilde{h}_{m - \\frac{1}{2}, n - \\frac{1}{2}, p - \\frac{1}{2}} =
     [\\hat{\\nabla} \\times \\hat{g}]_{m - \\frac{1}{2}, n - \\frac{1}{2}, p - \\frac{1}{2}} $$

  where $\\hat{g}$ and $\\tilde{g}$ are located at $(m,n,p)$
  with components at $(m \\pm \\frac{1}{2},n,p)$ etc.,
  while $\\hat{h}$ and $\\tilde{h}$ are located at $(m \\pm \\frac{1}{2}, n \\pm \\frac{1}{2}, p \\pm \\frac{1}{2})$
  with components at $(m, n \\pm \\frac{1}{2}, p \\pm \\frac{1}{2})$ etc.


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
        |/_x                  :....||.<.....|  (m+1, n+1, p+1/2)
                              /    ||      /
                           | v     ||   | ^
                           |/           |/
             (m, n, p+1/2) |_____>______|  (m+1, n, p+1/2)



Maxwell's Equations
===================

If we discretize both space (m,n,p) and time (l), Maxwell's equations become

 $$ \\begin{aligned}
  \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}} &= -\\tilde{\\partial}_t \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}
                                                                         - \\hat{M}_{l, \\vec{r} + \\frac{1}{2}}  \\\\
  \\hat{\\nabla} \\times \\hat{H}_{l-\\frac{1}{2},\\vec{r} + \\frac{1}{2}} &= \\hat{\\partial}_t \\tilde{D}_{l, \\vec{r}}
                                                                             + \\tilde{J}_{l-\\frac{1}{2},\\vec{r}} \\\\
  \\tilde{\\nabla} \\cdot \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}} &= 0 \\\\
  \\hat{\\nabla} \\cdot \\tilde{D}_{l,\\vec{r}} &= \\rho_{l,\\vec{r}}
 \\end{aligned} $$

 with

 $$ \\begin{aligned}
  \\hat{B}_{\\vec{r}} &= \\mu_{\\vec{r} + \\frac{1}{2}} \\cdot \\hat{H}_{\\vec{r} + \\frac{1}{2}} \\\\
  \\tilde{D}_{\\vec{r}} &= \\epsilon_{\\vec{r}} \\cdot \\tilde{E}_{\\vec{r}}
 \\end{aligned} $$

where the spatial subscripts are abbreviated as $\\vec{r} = (m, n, p)$ and
$\\vec{r} + \\frac{1}{2} = (m + \\frac{1}{2}, n + \\frac{1}{2}, p + \\frac{1}{2})$,
$\\tilde{E}$ and $\\hat{H}$ are the electric and magnetic fields,
$\\tilde{J}$ and $\\hat{M}$ are the electric and magnetic current distributions,
and $\\epsilon$ and $\\mu$ are the dielectric permittivity and magnetic permeability.

The above is Yee's algorithm, written in a form analogous to Maxwell's equations.
The time derivatives can be expanded to form the update equations:

    [code: Maxwell's equations updates]
    H[i, j, k] -= dt * (curl_forward(E)[i, j, k] + M[t, i, j, k]) /      mu[i, j, k]
    E[i, j, k] += dt * (curl_back(   H)[i, j, k] + J[t, i, j, k]) / epsilon[i, j, k]

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

Taking the backward curl of the $\\tilde{\\nabla} \\times \\tilde{E}$ equation and
replacing the resulting $\\hat{\\nabla} \\times \\hat{H}$ term using its respective equation,
and setting $\\hat{M}$ to zero, we can form the discrete wave equation:

$$
  \\begin{aligned}
  \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}} &=
      -\\tilde{\\partial}_t \\hat{B}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}
                          - \\hat{M}_{l-1, \\vec{r} + \\frac{1}{2}}  \\\\
  \\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}} &=
    -\\tilde{\\partial}_t \\hat{H}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}  \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}}) &=
    \\hat{\\nabla} \\times (-\\tilde{\\partial}_t \\hat{H}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}})  \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}}) &=
    -\\tilde{\\partial}_t \\hat{\\nabla} \\times \\hat{H}_{l-\\frac{1}{2}, \\vec{r} + \\frac{1}{2}}  \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}}) &=
    -\\tilde{\\partial}_t \\hat{\\partial}_t \\epsilon_{\\vec{r}} \\tilde{E}_{l, \\vec{r}} + \\hat{\\partial}_t \\tilde{J}_{l-\\frac{1}{2},\\vec{r}} \\\\
  \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{l,\\vec{r}})
            + \\tilde{\\partial}_t \\hat{\\partial}_t \\epsilon_{\\vec{r}} \\cdot \\tilde{E}_{l, \\vec{r}}
            &= \\tilde{\\partial}_t \\tilde{J}_{l - \\frac{1}{2}, \\vec{r}}
  \\end{aligned}
$$


Frequency domain
----------------

We can substitute in a time-harmonic fields

$$
 \\begin{aligned}
 \\tilde{E}_{l, \\vec{r}} &= \\tilde{E}_{\\vec{r}} e^{-\\imath \\omega l \\Delta_t} \\\\
 \\tilde{J}_{l, \\vec{r}} &= \\tilde{J}_{\\vec{r}} e^{-\\imath \\omega (l - \\frac{1}{2}) \\Delta_t}
 \\end{aligned}
$$

resulting in

$$
 \\begin{aligned}
 \\tilde{\\partial}_t &\\Rightarrow (e^{ \\imath \\omega \\Delta_t} - 1) / \\Delta_t = \\frac{-2 \\imath}{\\Delta_t} \\sin(\\omega \\Delta_t / 2) e^{-\\imath \\omega \\Delta_t / 2} = -\\imath \\Omega e^{-\\imath \\omega \\Delta_t / 2}\\\\
   \\hat{\\partial}_t &\\Rightarrow (1 - e^{-\\imath \\omega \\Delta_t}) / \\Delta_t = \\frac{-2 \\imath}{\\Delta_t} \\sin(\\omega \\Delta_t / 2) e^{ \\imath \\omega \\Delta_t / 2} = -\\imath \\Omega e^{ \\imath \\omega \\Delta_t / 2}\\\\
 \\Omega &= 2 \\sin(\\omega \\Delta_t / 2) / \\Delta_t
 \\end{aligned}
$$

This gives the frequency-domain wave equation,

$$
 \\hat{\\nabla} \\times (\\mu^{-1}_{\\vec{r} + \\frac{1}{2}} \\cdot \\tilde{\\nabla} \\times \\tilde{E}_{\\vec{r}})
    -\\Omega^2 \\epsilon_{\\vec{r}} \\cdot \\tilde{E}_{\\vec{r}} = \\imath \\Omega \\tilde{J}_{\\vec{r}}
$$


Plane waves and Dispersion relation
------------------------------------

With uniform material distribution and no sources

$$
 \\begin{aligned}
 \\mu_{\\vec{r} + \\frac{1}{2}} &= \\mu \\\\
 \\epsilon_{\\vec{r}} &= \\epsilon \\\\
 \\tilde{J}_{\\vec{r}} &= 0 \\\\
 \\end{aligned}
$$

the frequency domain wave equation simplifies to

$$ \\hat{\\nabla} \\times \\tilde{\\nabla} \\times \\tilde{E}_{\\vec{r}} - \\Omega^2 \\epsilon \\mu \\tilde{E}_{\\vec{r}} = 0 $$

Since $\\hat{\\nabla} \\cdot \\tilde{E}_{\\vec{r}} = 0$, we can simplify

$$
 \\begin{aligned}
 \\hat{\\nabla} \\times \\tilde{\\nabla} \\times \\tilde{E}_{\\vec{r}}
   &= \\tilde{\\nabla}(\\hat{\\nabla} \\cdot \\tilde{E}_{\\vec{r}}) - \\hat{\\nabla} \\cdot \\tilde{\\nabla} \\tilde{E}_{\\vec{r}} \\\\
   &= - \\hat{\\nabla} \\cdot \\tilde{\\nabla} \\tilde{E}_{\\vec{r}} \\\\
   &= - \\tilde{\\nabla}^2 \\tilde{E}_{\\vec{r}}
 \\end{aligned}
$$

and we get

$$  \\tilde{\\nabla}^2 \\tilde{E}_{\\vec{r}} + \\Omega^2 \\epsilon \\mu \\tilde{E}_{\\vec{r}} = 0 $$

We can convert this to three scalar-wave equations of the form

$$ (\\tilde{\\nabla}^2 + K^2) \\phi_{\\vec{r}} = 0 $$

with $K^2 = \\Omega^2 \\mu \\epsilon$. Now we let

$$  \\phi_{\\vec{r}} = A e^{\\imath (k_x m \\Delta_x + k_y n \\Delta_y + k_z p \\Delta_z)}  $$

resulting in

$$
 \\begin{aligned}
 \\tilde{\\partial}_x &\\Rightarrow (e^{ \\imath k_x \\Delta_x} - 1) / \\Delta_t = \\frac{-2 \\imath}{\\Delta_x} \\sin(k_x \\Delta_x / 2) e^{ \\imath k_x \\Delta_x / 2} = \\imath K_x e^{ \\imath k_x \\Delta_x / 2}\\\\
   \\hat{\\partial}_x &\\Rightarrow (1 - e^{-\\imath k_x \\Delta_x}) / \\Delta_t = \\frac{-2 \\imath}{\\Delta_x} \\sin(k_x \\Delta_x / 2) e^{-\\imath k_x \\Delta_x / 2} = \\imath K_x e^{-\\imath k_x \\Delta_x / 2}\\\\
 K_x &= 2 \\sin(k_x \\Delta_x / 2) / \\Delta_x \\\\
 \\end{aligned}
$$

with similar expressions for the y and z dimnsions (and $K_y, K_z$).

This implies

$$
  \\tilde{\\nabla}^2 = -(K_x^2 + K_y^2 + K_z^2) \\phi_{\\vec{r}} \\\\
  K_x^2 + K_y^2 + K_z^2 = \\Omega^2 \\mu \\epsilon = \\Omega^2 / c^2
$$

where $c = \\sqrt{\\mu \\epsilon}$.

Assuming real $(k_x, k_y, k_z), \\omega$ will be real only if

$$ c^2 \\Delta_t^2 = \\frac{\\Delta_t^2}{\\mu \\epsilon} < 1/(\\frac{1}{\\Delta_x^2} + \\frac{1}{\\Delta_y^2} + \\frac{1}{\\Delta_z^2}) $$

If $\\Delta_x = \\Delta_y = \\Delta_z$, this simplifies to $c \\Delta_t < \\Delta_x / \\sqrt{3}$.
This last form can be interpreted as enforcing causality; the distance that light
travels in one timestep (i.e., $c \\Delta_t$) must be less than the diagonal
of the smallest cell ( $\\Delta_x / \\sqrt{3}$ when on a uniform cubic grid).


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

Place the E fore-vectors at integer indices $r = (m, n, p)$ and the H back-vectors
at fractional indices $r + \\frac{1}{2} = (m + \\frac{1}{2}, n + \\frac{1}{2},
p + \\frac{1}{2})$. Remember that these are indices and not coordinates; they can
correspond to arbitrary (monotonically increasing) coordinates depending on the cell widths.

Draw lines to denote the planes on which the H components and back-vectors are defined.
For simplicity, don't draw the equivalent planes for the E components and fore-vectors,
except as necessary to show their locations -- it's easiest to just connect them to their
associated H-equivalents.

The result looks something like this:

    [figure: Component centers]
                                                                 p=
              [H]__________Hx___________[H]_____Hx______[H]   __ +1/2
      z y     /:           /:           /:      /:      /|     |      |
      |/_x   / :          / :          / :     / :     / |     |      |
            /  :         /  :         /  :    /  :    /  |     |      |
          Hy   :       Ez...........Hy   :  Ez......Hy   |     |      |
          /:   :        :   :       /:   :   :   :  /|   |     |      |
         / :  Hz        :  Ey....../.:..Hz   :  Ey./.|..Hz    __ 0    | dz[0]
        /  :  /:        :  /      /  :  /:   :  / /  |  /|     |      |
       /_________________________/_______________/   | / |     |      |
       |   :/  :        :/       |   :/  :   :/  |   |/  |     |      |
       |  Ex   :       [E].......|..Ex   :  [E]..|..Ex   |     |      |
       |       :                 |       :       |       |     |      |
       |      [H]..........Hx....|......[H].....H|x.....[H]   __ --------- (n=+1/2, p=-1/2)
       |      /                  |      /        |      /     /       /
      Hz     /                  Hz     /        Hz     /     /       /
       |    /                    |    /          |    /     /       /
       |  Hy                     |  Hy           |  Hy    __ 0     / dy[0]
       |  /                      |  /            |  /     /       /
       | /                       | /             | /     /       /
       |/                        |/              |/     /       /
      [H]__________Hx___________[H]_____Hx______[H]   __ -1/2  /
                                                           =n
       |------------|------------|-------|-------|
     -1/2           0          +1/2     +1     +3/2 = m

        ------------------------- ----------------
                  dx[0]                  dx[1]

      Part of a nonuniform "base grid", with labels specifying
      positions of the various field components. [E] fore-vectors
      are at the cell centers, and [H] back-vectors are at the
      vertices. H components along the near (-y) top (+z) edge
      have been omitted to make the insides of the cubes easier
      to visualize.

The above figure shows where all the components are located; however, it is also useful to show
what volumes those components correspond to. Consider the Ex component at `m = +1/2`: it is
shifted in the x-direction by a half-cell from the E fore-vector at `m = 0` (labeled `[E]`
in the figure). It corresponds to a volume between `m = 0` and `m = +1` (the other
dimensions are not shifted, i.e. they are still bounded by `n, p = +-1/2`). (See figure
below). Since `m` is an index and not an x-coordinate, the Ex component is not necessarily
at the center of the volume it represents, and the x-length of its volume is the derived
quantity `dx'[0] = (dx[0] + dx[1]) / 2` rather than the base `dx`.
(See also `Scalar derivatives and cell shifts`).

    [figure: Ex volumes]
                                                                 p=
               <_________________________________________>   __ +1/2
      z y     <<           /:           /       /:      >>    |      |
      |/_x   < <          / :          /       / :     > >    |      |
            <  <         /  :         /       /  :    >  >    |      |
           <   <        /   :        /       /   :   >   >    |      |
          <:   <       /    :        :      /    :  >:   >    |      |
         < :   <      /     :        :     /     : > :   >   __ 0    | dz[0]
        <  :   <     /      :        :    /      :>  :   >    |      |
       <____________/____________________/_______>   :   >    |      |
       <   :   <    |       :        :   |       >   :   >    |      |
       <  Ex   <    |       :       Ex   |       >  Ex   >    |      |
       <   :   <    |       :        :   |       >   :   >    |      |
       <   :   <....|.......:........:...|.......>...:...>   __ --------- (n=+1/2, p=-1/2)
       <   :  <     |      /         :  /|      />   :  >    /       /
       <   : <      |     /          : / |     / >   : >    /       /
       <   :<       |    /           :/  |    /  >   :>    /       /
       <   <        |   /            :   |   /   >   >    _ 0     / dy[0]
       <  <         |  /                 |  /    >  >    /       /
       < <          | /                  | /     > >    /       /
       <<           |/                   |/      >>    /       /
       <____________|____________________|_______>   __ -1/2  /
                                                         =n
       |------------|------------|-------|-------|
     -1/2           0          +1/2      +1    +3/2 = m

       ~------------ -------------------- -------~
         dx'[-1]          dx'[0]           dx'[1]

     The Ex values are positioned on the x-faces of the base
     grid. They represent the Ex field in volumes shifted by
     a half-cell in the x-dimension, as shown here. Only the
     center cell (with width dx'[0]) is fully shown; the
     other two are truncated (shown using >< markers).

     Note that the Ex positions are the in the same positions
     as the previous figure; only the cell boundaries have moved.
     Also note that the points at which Ex is defined are not
     necessarily centered in the volumes they represent; non-
     uniform cell sizes result in off-center volumes like the
     center cell here.

The next figure shows the volumes corresponding to the Hy components, which
are shifted in two dimensions (x and z) compared to the base grid.

    [figure: Hy volumes]
                                                                 p=
      z y     mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm   __ +1/2  s
      |/_x   <<           m:                    m:      >>    |       |
            < <          m :                   m :     > >    |       | dz'[1]
           <  <         m  :                  m  :    >  >    |       |
         Hy........... m........Hy...........m......Hy   >    |       |
         <    <       m    :                m    :  >    >    |       |
        <     ______ m_____:_______________m_____:_>______   __ 0
       <      <     m     /:              m     / >      >    |       |
      mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm       >    |       |
      <       <    |    /  :             |    /  >       >    |       | dz'[0]
      <       <    |   /   :             |   /   >       >    |       |
      <       <    |  /    :             |  /    >       >    |       |
      <       wwwww|w/wwwwwwwwwwwwwwwwwww|w/wwwww>wwwwwwww   __       s
      <      <     |/     w              |/     w>      >    /         /
      _____________|_____________________|________     >    /         /
      <    <       |    w                |    w  >    >    /         /
      <  Hy........|...w........Hy.......|...w...>..Hy    _ 0       / dy[0]
      < <          |  w                  |  w    >  >    /         /
      <<           | w                   | w     > >    /         /
      <            |w                    |w      >>    /         /
      wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww   __ -1/2    /

      |------------|------------|--------|-------|
    -1/2           0          +1/2      +1     +3/2 = m

      ~------------ --------------------- -------~
         dx'[-1]            dx'[0]         dx'[1]

     The Hy values are positioned on the y-edges of the base
     grid. Again here, the 'Hy' labels represent the same points
     as in the basic grid figure above; the edges have shifted
     by a half-cell along the x- and z-axes.

     The grid lines _|:/ are edges of the area represented by
     each Hy value, and the lines drawn using <m>.w represent
     edges where a cell's faces extend beyond the drawn area
     (i.e. where the drawing is truncated in the x- or z-
     directions).


Datastructure: dx_lists_t
-------------------

In this documentation, the E fore-vectors are placed on the base grid. An
equivalent formulation could place the H back-vectors on the base grid instead.
However, in the case of a non-uniform grid, the operation to get from the "base"
cell widths to "derived" ones is not its own inverse.

The base grid's cell sizes could be fully described by a list of three 1D arrays,
specifying the cell widths along all three axes:

    [dx, dy, dz] = [[dx[0], dx[1], ...], [dy[0], ...], [dz[0], ...]]

Note that this is a list-of-arrays rather than a 2D array, as the simulation domain
may have a different number of cells along each axis.

Knowing the base grid's cell widths and the boundary conditions (periodic unless
otherwise noted) is enough information to calculate the cell widths  `dx'`, `dy'`,
and `dz'` for the derived grids.

However, since most operations are trivially generalized to allow either E or H
to be defined on the base grid, they are written to take the a full set of base
and derived cell widths, distinguished by which field they apply to rather than
their "base" or "derived" status. This removes the need for each function to
generate the derived widths, and makes the "base" vs "derived" distinction
unnecessary in the code.

The resulting data structure containing all the cell widths takes the form of a
list-of-lists-of-arrays. The first list-of-arrays provides the cell widths for
the E-field fore-vectors, while the second list-of-arrays does the same for the
H-field back-vectors:

     [[[dx_e[0], dx_e[1], ...], [dy_e[0], ...], [dz_e[0], ...]],
      [[dx_h[0], dx_h[1], ...], [dy_h[0], ...], [dz_h[0], ...]]]

   where `dx_e[0]` is the x-width of the `m=0` cells, as used when calculating dE/dx,
   and `dy_h[0]` is  the y-width of the `n=0` cells, as used when calculating dH/dy, etc.


Permittivity and Permeability
=============================

Since each vector component of E and H is defined in a different location and represents
a different volume, the value of the spatially-discrete `epsilon` and `mu` can also be
different for all three field components, even when representing a simple planar interface
between two isotropic materials.

As a result, `epsilon` and `mu` are taken to have the same dimensions as the field, and
composed of the three diagonal tensor components:

    [equations: epsilon_and_mu]
    epsilon = [epsilon_xx, epsilon_yy, epsilon_zz]
    mu = [mu_xx, mu_yy, mu_zz]

or

$$
 \\epsilon = \\begin{bmatrix} \\epsilon_{xx} & 0 & 0 \\\\
                              0 & \\epsilon_{yy} & 0 \\\\
                              0 & 0 & \\epsilon_{zz} \\end{bmatrix}
$$
$$
 \\mu = \\begin{bmatrix} \\mu_{xx} & 0 & 0 \\\\
                         0 & \\mu_{yy} & 0 \\\\
                         0 & 0 & \\mu_{zz} \\end{bmatrix}
$$

where the off-diagonal terms (e.g. `epsilon_xy`) are assumed to be zero.

High-accuracy volumetric integration of shapes on multiple grids can be performed
by the [gridlock](https://mpxd.net/code/jan/gridlock) module.

The values of the vacuum permittivity and permability effectively become scaling
factors that appear in several locations (e.g. between the E and H fields). In
order to limit floating-point inaccuracy and simplify calculations, they are often
set to 1 and relative permittivities and permeabilities are used in their places;
the true values can be multiplied back in after the simulation is complete if non-
normalized results are needed.
"""

from .types import fdfield_t, vfdfield_t, cfdfield_t, vcfdfield_t, dx_lists_t, dx_lists_mut
from .types import fdfield_updater_t, cfdfield_updater_t
from .vectorization import vec, unvec
from . import operators, functional, types, vectorization

