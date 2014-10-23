---
---

# Conventions

Here, we carefully examine all the assumptions built in to the
conventions for the `spherical_functions` package, and relate these
choices to those made by other authors.

## Quaternions, rotations, spherical coordinates

We first establish our basis vectors: $(\basis{x}, \basis{y},
\basis{z})$.  These remain fixed at all times.

A unit quaternion (or "rotor") $\rotor{R}$ can rotate a vector
$\vec{v}$ into a new vector $\Rotated{\vec{v}}$ according to the
formula
\begin{equation}
  \Rotated{\vec{v}} = \rotor{R}\, \vec{v}\, \rotor{R}^{-1}.
\end{equation}
In principle, a unit quaternion obeys $\co{\rotor{R}} =
\rotor{R}^{-1}$.  In practice, however, the system is more stable
numerically if we explicitly use the inversion.

![Spherical-coordinate system; By Andeggs, via Wikimedia Commons]({{ site.url }}/images/3D_Spherical_Coords.svg){: style="float:right;height:200px"}

We will use the standard spherical coordinates $(\vartheta, \varphi)$,
in the physics convention.  That is, $\vartheta$ measures the angle
from the positive $\basis{z}$ axis and $\varphi$ measures the angle of
the projection into the $\basis{x}$-$\basis{y}$ plane from the
positive $\basis{x}$ axis, as shown in the figure.  Now, if $\hat{n}$
is the unit vector in that direction, we construct a rotor
\begin{equation}
  \rthetaphi = e^{\varphi \basis{z}/2}\, e^{\vartheta \basis{y}/2}.
\end{equation}

This rotor is particularly useful, because $\hat{n}$ and the two
standard tangent vectors at that point are all given by rotations of
the basis vectors $(\basis{x}, \basis{y}, \basis{z})$.  Specifically,
we have
\begin{align}
  \hat{\vartheta} &= \rthetaphi \basis{x} \rthetaphi^{-1}, \\\\
  \hat{\varphi} &= \rthetaphi \basis{y} \rthetaphi^{-1}, \\\\
  \hat{n} &= \rthetaphi \basis{z} \rthetaphi^{-1}.
\end{align}


### Mathematica

In the "Applications" section of Mathematica's documentation page for
`WignerD`, the rotation matrix is constructed from Euler angles
$(\psi,\theta,\phi)$ according to the expression

```
RotationMatrix[-phi, {0, 0, 1}].RotationMatrix[-theta, {0, 1, 0}].RotationMatrix[-psi, {0, 0, 1}]
```

This is the inverse of the matrix given by

```
RotationMatrix[psi, {0, 0, 1}].RotationMatrix[theta, {0, 1, 0}].RotationMatrix[phi, {0, 0, 1}]
```

The latter, of course, would be equivalent to a rotor
$e^{\psi\basis{z}/2}\, e^{\theta\basis{y}/2}\, e^{\phi\basis{z}/2}$,
which is what I would have denoted $\rotor{R}_{(\psi, \theta,
\phi)}$.  So my rotor gives the inverse rotation of Mathematica's
Euler angles.  This could also be viewed as a disagreement over active
and passive transformations.


The same page states that

```
WignerD[{j,m1,m2},psi,theta,phi]
```

gives the function $\mathfrak{D}^j_{m_1,m_2}(\psi,\theta,\phi)$, and
the spherical harmonics are related to $\mathfrak{D}$ by

\begin{equation}
  \mathfrak{D}^\ell\_{0,m}(0, \theta, \phi) =
  \sqrt{\frac{4\pi}{2\ell+1}} Y\_{\ell,m} (\theta, \phi).
\end{equation}
