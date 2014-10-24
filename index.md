---
---

# Conventions

Defined exclusively in terms of quaternions, the Wigner $\mathfrak{D}$
matrices and spherical-harmonic functions (spin-weighted and scalar)
are pretty simple, and easy to make internally consistent.  However,
it is important to establish which conventions are in use ---
especially in comparison to other source.  Here, I carefully examine
all the assumptions built in to the conventions for the
`spherical_functions` package, and relate these choices to those made
by other authors.

## Quaternions, rotations, spherical coordinates

A unit quaternion (or "rotor") $\rotor{R}$ can rotate a vector
$\vec{v}$ into a new vector $\Rotated{\vec{v}}$ according to the
formula
\begin{equation}
  \Rotated{\vec{v}} = \rotor{R}\, \vec{v}\, \rotor{R}^{-1}.
\end{equation}
In principle, a unit quaternion obeys $\co{\rotor{R}} =
\rotor{R}^{-1}$.  In practice, however, there are cases where the
system is (slightly slower, but) more stable numerically if the
explicit inversion is used.  And since the inversion is such a simple
operation, we simply use it.

The Wigner $\mathfrak{D}$ matrices are derived in terms of a
particular split of the quaternion group into two parts.  Explicitly,
a quaternion $\quat{Q}$ can be expressed in terms of two complex
numbers $\quat{Q}\_a = \quat{Q}\_1 + i\, \quat{Q}\_z$ and $\quat{Q}\_b
= \quat{Q}\_y + i\, \quat{Q}\_x$.<sup>[^1]</sup> This is only important because
it allows us to verify the multiplication law
\begin{align}
  (\quat{P}\,\quat{Q})\_a &= \quat{P}\_a\, \quat{Q}\_a - \co{\quat{P}}\_b\, \quat{Q}\_b, \\\\
  (\quat{P}\,\quat{Q})\_b &= \quat{P}\_b\, \quat{Q}\_a + \co{\quat{P}}\_a\, \quat{Q}\_b.
\end{align}
Given a rotor $\rotor{R}$, these two complex numbers are the
quantities actually used in computing the $\mathfrak{D}$ matrix
elements.

![Spherical-coordinate system; By Andeggs, via Wikimedia Commons]({{ site.url }}/spherical_functions/images/3D_Spherical_Coords.svg){: style="float:right;height:200px"}

When necessary to make contact with previous literature, we will use
the standard spherical coordinates $(\vartheta, \varphi)$, in the
physics convention.  To explain this, we first establish our
right-handed basis vectors: $(\basis{x}, \basis{y}, \basis{z})$, which
remain fixed at all times.  Thus, $\vartheta$ measures the angle from
the positive $\basis{z}$ axis and $\varphi$ measures the angle of the
projection into the $\basis{x}$-$\basis{y}$ plane from the positive
$\basis{x}$ axis, as shown in the figure.

Now, if $\hat{n}$ is the unit vector in that direction, we construct a
related rotor
\begin{equation}
  \rthetaphi = e^{\varphi \basis{z}/2}\, e^{\vartheta \basis{y}/2}.
\end{equation}
This can be obtained as an `np.quaternion` object in python as

```python
>>> import numpy as np, quaternion
>>> vartheta, varphi = 0.1, 0.2
>>> R_tp = np.quaternion.from_spherical_coords(vartheta, varphi)
```

Here, rotations are given assuming the right-hand screw rule, so that
this corresponds to an initial rotation through the angle $\vartheta$
about $\basis{y}$, followed by a rotation through $\varphi$ about
$\basis{z}$.  (The factors of $1/2$ are present because $\rthetaphi$
is essentially the square-root of the rotation; the effect of this
rotor produces the full rotations through $\vartheta$ and $\varphi$.)
This rotor is particularly useful, because $\hat{n}$ and the two
standard tangent vectors at that point are all given by rotations of
the basis vectors $(\basis{x}, \basis{y}, \basis{z})$.  Specifically,
we have
\begin{align}
  \hat{\vartheta} &= \rthetaphi\,\basis{x}\,\rthetaphi^{-1}, \\\\
  \hat{\varphi} &= \rthetaphi\, \basis{y}\, \rthetaphi^{-1}, \\\\
  \hat{n} &= \rthetaphi\, \basis{z}\, \rthetaphi^{-1}.
\end{align}
Note that we interpret the rotations in the active sense, where a
point moves, while our coordinate system and basis vectors are left
fixed.  Hopefully, this won't cause too much confusion if we express
the original point with the vector $\basis{z}$, for example, and the
rotated point is therefore given by some $\rotor{R}\, \basis{z}\,
\rotor{R}$.

### Euler angles

Euler angles are a terrible set of coordinates for the rotation group.
Compared to the other three standard presentations of rotations
(rotation matrices, axis-angle form, and the closely related unit
quaternions), Euler angles present no advantages and many severe
disadvantages.  Composition of rotations is complicated, numerically
slow and inaccurate, and essentially requires transformation to a
different presentation anyway.  Interpolation of Euler angles is
meaningless and prone to severe distortions.  Most damningly of all
are the coordinate singularities (gimbal lock).  To summarize, Euler
angles are absolutely --- and by a wide margin --- the worst way to
deal with rotations.

We can work entirely without Euler angles.  (My own work simply never
uses them; the $\mathfrak{D}$ matrices are written directly in terms
of rotors.)  Nonetheless, they are ubiquitous throughout the
literature.  And while we can work entirely without Euler angles.  So,
to make contact with that literature, we will need to choose a
convention for constructing a rotation from a triple of angles
$(\alpha, \beta, \gamma)$.  We therefore define the rotor
\begin{equation}
  R\_{(\alpha, \beta, \gamma)} = e^{\alpha\, \basis{z}/2}\, e^{\beta\,
  \basis{y}/2}\, e^{\gamma\, \basis{z}/2}.
\end{equation}
This can be obtained as an `np.quaternion` object in python as

```python
>>> import numpy as np, quaternion
>>> alpha, beta, gamma = 0.1, 0.2, 0.3
>>> R_euler = np.quaternion.from_euler_angles(alpha, beta, gamma)
```

Note that the rotations are always taken about the *fixed* axes
$\basis{z}$ and $\basis{y}$.  Also, this is in operator form, so it
must be read from right to left: The rotation is given by an initial
rotation through $\gamma$ about the $\basis{z}$ axis, followed by a
rotation through $\beta$ about the $\basis{y}$ axis, followed by a
final rotation through $\alpha$ about the $\basis{z}$ axis.  This may
seem slightly backwards, but it is a common convention --- in
particular, it is the one adopted by Wikipedia in its
[Wigner-D article](https://en.wikipedia.org/wiki/Wigner_D-matrix#Definition_of_the_Wigner_D-matrix).

It is worth noting that the standard right-handed basis vectors
$(\basis{x}, \basis{y}, \basis{z})$ can be identified with generators
of rotations usually seen in quantum mechanics (or generally just
special-function theory) according to the rule
\begin{align}
  \frac{\basis{x}}{2} &\mapsto -i\, J\_x, \\\\
  \frac{\basis{y}}{2} &\mapsto -i\, J\_y, \\\\
  \frac{\basis{z}}{2} &\mapsto -i\, J\_z.
\end{align}
This is important when relating quaternion expressions to expressions
more commonly seen in the literature.  In particular, with this
identification, we have the usual commutation relations
\begin{align}
  \left[\frac{\basis{x}}{2}, \frac{\basis{y}}{2}\right] = \frac{\basis{z}}{2} &\mapsto
  [J\_x, J\_y] = i\, J\_z, \\\\
  \left[\frac{\basis{y}}{2}, \frac{\basis{z}}{2}\right] = \frac{\basis{x}}{2} &\mapsto
  [J\_y, J\_z] = i\, J\_x, \\\\
  \left[\frac{\basis{z}}{2}, \frac{\basis{x}}{2}\right] = \frac{\basis{y}}{2} &\mapsto
  [J\_z, J\_x] = i\, J\_y.
\end{align}
And in any case, this certainly clarifies what to do with expressions
like the following from Wikipedia:
\begin{equation}
  \mathcal{R}(\alpha,\beta,\gamma) = e^{-i\alpha\, J\_z}\,
  e^{-i\beta\, J\_y} e^{-i\gamma\, J\_z},
\end{equation}
which shows that my interpretation of Euler angles is the same as
Wikipedia's.

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

However, there are still other disagreements.  The same page states
that

```
WignerD[{j,m1,m2},psi,theta,phi]
```

gives the function $\mathfrak{D}^j\_{m\_1,m\_2}(\psi,\theta,\phi)$, and
the spherical harmonics are related to $\mathfrak{D}$ by

\begin{equation}
  \mathfrak{D}^\ell\_{0,m}(0, \theta, \phi) =
  \sqrt{\frac{4\pi}{2\ell+1}} Y\_{\ell,m} (\theta, \phi).
\end{equation}



[^1]: This, of course, is not a productive way of *thinking about*
      quaternions, but it is a very useful way of calculating with
      them.  That is, this decomposition into two complex components
      is something that the user probably does not need to worry
      about.  It is, however, an isomorphism between quaternions and
      the usual (awful) presentation of Pauli spinors as two-component
      complex vectors.
