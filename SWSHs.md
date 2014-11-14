---
---

# Spin-weighted spherical harmonics

Spin-weighted spherical harmonics (SWSHs) generalize the standard
spherical-harmonic functions.  In particular, there is a spin weight
$s$ associated with each class of SWSHs ${}_{s}Y\_{\ell,m}$, and $s=0$
corresponds to the standard spherical harmonics $Y\_{\ell,m}$.  They
can be thought of as special cases of the
[Wigner $\mathfrak{D}$ matrices](WignerDMatrices.html).  This allows
us to use the representation property of the $\mathfrak{D}$ matrices
to understand how SWSHs rotate.  More importantly, if a spin-weighted
function is expanded in a basis of SWSHs, we can derive a formula for
the coefficients of that expansion with respect to a rotated frame.
These properties will all be derived and discussed below.

This discussion is essentially an updated version of the one found in
[my paper](http://arxiv.org/abs/1302.2919) of a couple years ago.  For
more references, see the citations therein.

## Spin-weighted functions

Spin-weighted functions on the sphere are unusual objects.  They take
values in the complex numbers $\mathbb{C}$.  One would be forgiven,
therefore, for thinking that the field's value at a point is a scalar.
The key property of a scalar is its trivial transformation law, which
spin-weighted functions do not obey.[^1] To be more precise, imagine
we have some function ${}\_{s}f(\boldsymbol{n})$, where
$\boldsymbol{n}$ is the vector from the origin to the corresponding
point on the sphere, and $s$ is the spin weight.  Now, if we rotate
our basis, we have a new function ${}\_{s}f'$; the same rotation gives
us a new field point $\boldsymbol{n}'$.  A scalar function obeys
\begin{equation}
  \label{eq:ScalarTransformation}
  {}\_{0}f'(\boldsymbol{n}') = {}\_{0}f(\boldsymbol{n})
\end{equation}
On the other hand, a spin-weighted function obeys the somewhat
stranger transformation law
\begin{equation}
  \label{eq:SpinWeightedTransformation}
  {}\_{s}f'(\boldsymbol{n}') = {}\_{s}f(\boldsymbol{n})\, e^{-i\, s\, \gamma}
\end{equation}
for some angle $\gamma$.  The most important example of this
transformation is a rotation about the $\boldsymbol{n}$ vector (in the
*positive*, *right-handed* sense) through an angle $\gamma$.[^2]

It may be surprising that a transformation for which $\boldsymbol{n}'
= \boldsymbol{n}$ actually changes the value of a spin-weighted
function.  The reason for this strange behavior is that spin-weighted
functions are actually contractions of tensors with various terms of
the form $(\boldsymbol{\vartheta} + i\,
\boldsymbol{\varphi})/\sqrt{2}$, or the conjugate of that term.  In
this sense, the spin-weighted functions are actually tensor components
with respect to some coordinate basis.  So it makes sense that the
components should transform under coordinate transformations.

## Different ways of thinking about spin-weighted functions

It seems a little inconsistent that we should write spin-weighted
functions as functions of $\boldsymbol{n}$ alone; strictly speaking,
it seems like it would make more sense to write
${}\_{s}f(\boldsymbol{n}, \boldsymbol{\vartheta},
\boldsymbol{\varphi})$.  In fact, we can do a little better.  Those
three vectors are a little redundant; all the information provided by
them is actually carried in a single rotation operator:
\begin{align\*}
  \boldsymbol{\vartheta} &= \mathcal{R} \\{\basis{x}\\}, \\\\
  \boldsymbol{\varphi} &= \mathcal{R} \\{\basis{y}\\}, \\\\
  \boldsymbol{n} &= \mathcal{R} \\{\basis{z}\\}.
\end{align\*}
And so we can consider the spin-weighted functions not to be functions
of points on the sphere, but functions of rotation operators.
Given that the group of unit quaternions (rotors) is so vastly
preferable as a way of representing rotations, we will frequently
write spin-weighted functions as functions of a rotor
${}\_{s}f(\quat{R})$.

Here, the relation to the standard way of writing functions on the
sphere is clear.  ${}\_{s}f(\vartheta, \varphi)$ is the function's
value at the point given by coordinates $(\vartheta, \varphi)$.  But
for spin-weighted functions, there is an implicit dependence on the
standard tangent basis at that point, $(\boldsymbol{\vartheta},
\boldsymbol{\varphi})$.  By writing the spin-weighted function as a
function of the rotor $\quat{R}$, we are simultaneously

 1. encoding the point $(\vartheta, \varphi)$ as $\quat{R}\,
    \basis{z}\, \quat{R}^{-1}$, and
 2. encoding the dependence on the tangent basis as $\quat{R}\,
    \basis{x}\, \quat{R}^{-1}$ and $\quat{R}\, \basis{y}\,
    \quat{R}^{-1}$.

Moreover, this will make the relationship between SWSHs and the Wigner
$\mathfrak{D}$ matrices very simple, and remove apparent singularities
associated with the coordinates $(\vartheta, \varphi)$.  (In fact,
this approach will also be very helpful with the more general
Newman-Penrose formalism and Lorentz transformations.)

As a side note, a common mathematical device for simultaneously
recording a position on the sphere and the orientation of the tangent
basis is called the
["Hopf fibration"](http://en.wikipedia.org/wiki/Hopf_fibration).  This
is a fiber bundle, where the base space is the sphere $S^2$
(representing positions on the sphere), and the fibers are circles
$S^1$ (corresponding to orientations of the tangent basis at that
point).  The total space is simply $S^3$.  This, not coincidentally,
is also the space of unit quaternions.  So we see that the Hopf
fibration is just another way of looking at rotors.


## SWSHs in terms of Wigner's $\mathfrak{D}$ matrices

It is important to make contact with other sources in the literature,
to ensure that we use the same conventions---or at least understand
how our conventions differ.  First, we note that
[as shown before](index.html), the standard spherical coordinates
$(\vartheta, \varphi)$ and the corresponding standard tangent basis
vectors can be expressed simply in the form of the rotor
\begin{equation\*}
  \rthetaphi = e^{\varphi \basis{z}/2}\, e^{\vartheta \basis{y}/2}.
\end{equation\*}
Or, to put this in a form more useful for us, we have
\begin{equation\*}
  \quat{R}\_a = e^{i\,\varphi/2}\, \cos\frac{\beta}{2},
  \qquad
  \quat{R}\_b = e^{-i\,\varphi/2}\, \sin\frac{\beta}{2}.
\end{equation\*}
Plugging these expressions into the expression for $\mathfrak{D}$
given [here](WignerDMatrices.html#mjx-eqn-eqD_RaGeqRb), for example,
and comparing to the expressions in
[the data-formats paper](http://arxiv.org/abs/0709.0093), we see that
SWSHs can be written as
\begin{equation}
  \label{eq:SWSHFromWignerD}
  {}\_{s}Y\_{\ell,m} (\vartheta, \varphi)
  =
  {}\_{s}Y\_{\ell,m} (\quat{R}\_{(\vartheta, \varphi)})
  =
  (-1)^s\, \sqrt{\frac{2\ell+1} {4\pi}}\,
  \mathfrak{D}^{(\ell)}\_{m,-s} \left(\quat{R}\_{(\vartheta, \varphi)}
  \right).
\end{equation}


## Rotating SWSHs

Now, using Eq. \eqref{eq:SWSHFromWignerD} and the representation
property of Wigner's $\mathfrak{D}$ matrices, we can show how to
rotate the SWSH functions, and show the transformation of modes of a
decomposition of a general spin-weighted function by SWSHs.

We start of with the fact that the $\mathfrak{D}$ matrices form a
representation of the rotation group.  That is, given any two rotors
$\quat{R}\_1$ and $\quat{R}\_2$, we have
\begin{equation}
  \mathfrak{D}^{(\ell)}\_{m',m} (\quat{R}\_1\, \quat{R}\_2) =
  \sum\_{m''}
  \mathfrak{D}^{(\ell)}\_{m',m''} (\quat{R}\_1)\,
  \mathfrak{D}^{(\ell)}\_{m'',m} (\quat{R}\_2).
\end{equation}
From Eq. \eqref{eq:SWSHFromWignerD}, we see that it is possible to
rewrite this as
\begin{equation}
  {}\_{s}Y\_{\ell,m} (\quat{R}\_1\, \quat{R}\_2) =
  \sum\_{m'}
  \mathfrak{D}^{(\ell)}\_{m,m'} (\quat{R}\_1)\,
  {}\_{s}Y\_{\ell,m'}(\quat{R}\_2).
\end{equation}
In principle, this is the rotation law needed for SWSHs.

As we saw above, spin-weighted functions should be regarded as
functions on the rotation group, rather than as functions on points of
the sphere.  Unfortunately, this is not the standard treatment; a
canonical tangent basis is assumed from the coordinate system.  So we
need to write the arguments of the SWSHs in terms of spherical
coordinates.  But there's a wrinkle here.  If we assume that the
argument of the SWSH on the left-hand side above can be written as
spherical coordinates $\quat{R}\_{(\vartheta', \varphi')}$, the
argument of the SWSH on the right-hand side cannot be written purely
as spherical coordinates; there is some additional rotation needed.
In general, then, we must write the transformation in the form
\begin{equation}
  \quat{R}\_{(\vartheta', \varphi')} = \quat{R}\,
  \quat{R}\_{(\vartheta, \varphi)}\, e^{\gamma\, \basis{z}/2},
\end{equation}
for some angle $\gamma$, corresponding to a final rotation about the
vector joining the origin to the point at $(\vartheta', \varphi')$.
Fortunately, we can factor out this term using the known behavior of
spin-weighted functions:
\begin{align}
  {}\_{s}Y\_{\ell,m} (\quat{R}\_{(\vartheta', \varphi')}) &=
  \sum\_{m'}
  \mathfrak{D}^{(\ell)}\_{m,m'} (\quat{R})\,
  {}\_{s}Y\_{\ell,m'}(\quat{R}\_{(\vartheta, \varphi)}\, e^{\gamma\,
  \basis{z}/2})
  \nonumber \\\\ \label{eq:SWSHRotation}
  &= \sum\_{m'}
  \mathfrak{D}^{(\ell)}\_{m,m'} (\quat{R})\,
  {}\_{s}Y\_{\ell,m'}(\quat{R}\_{(\vartheta, \varphi)})\, e^{-i\, s\, \gamma}.
\end{align}
We see an interesting result from this: the spin-weighted spherical
harmonics do *not* transform among themselves under rotations---except
in the familiar case of $s=0$.

This curious factor of $e^{-i\, s\, \gamma}$, however, is precisely
what is needed to ensure that *modes* of an expansion in SWSHs *do*
transform among themselves under rotations.  To be more precise, let
us take a function ${}\_{s}f$, with a well defined spin weight $s$.
This can be expanded in the basis of SWSHs as
\begin{equation\*}
  {}\_{s}f(\quat{R}\_{(\vartheta, \varphi)}) = \sum\_m f^{\ell,m}
  {}\_{s}Y\_{\ell,m}(\quat{R}\_{(\vartheta, \varphi)}).
\end{equation\*}
Now, there is another field ${}\_{s}f'$ representing the physically
rotated field having the same magnitude at $(\vartheta', \varphi')$
as ${}\_{s}f$ has at $(\vartheta, \varphi)$.  This has a similar
expansion:
\begin{equation\*}
  {}\_{s}f'(\quat{R}\_{(\vartheta', \varphi')}) = \sum\_m f'^{\ell,m}
  {}\_{s}Y\_{\ell,m}(\quat{R}\_{(\vartheta', \varphi')}).
\end{equation\*}
Now, recalling Eq. \eqref{eq:SpinWeightedTransformation}
${}\_{s}f'(\quat{R}\_{(\vartheta', \varphi')}) =
{}\_{s}f(\quat{R}\_{(\vartheta, \varphi)})\, e^{-i\, s\, \gamma}$, we
can combine the last three equations.
{::comment}
\begin{align\*}
  {}\_{s}f'(\quat{R}\_{(\vartheta', \varphi')})
  &=
  \sum\_m f'^{\ell,m} {}\_{s}Y\_{\ell,m}(\quat{R}\_{(\vartheta', \varphi')})
  \\\\ &=
  \sum\_m f'^{\ell,m} \sum\_{m'}
  \mathfrak{D}^{(\ell)}\_{m,m'} (\quat{R})\,
  {}\_{s}Y\_{\ell,m'}(\quat{R}\_{(\vartheta, \varphi)})\, e^{-i\, s\, \gamma}
  \\\\ &=
  \sum\_{m',m} f'^{\ell,m'}
  \mathfrak{D}^{(\ell)}\_{m',m} (\quat{R})\,
  {}\_{s}Y\_{\ell,m}(\quat{R}\_{(\vartheta, \varphi)})\, e^{-i\, s\, \gamma}
  \\\\ &=
  {}\_{s}f(\quat{R}\_{(\vartheta, \varphi)})\, e^{-i\, s\, \gamma}
  \\\\ &=
  \sum\_m f^{\ell,m} {}\_{s}Y\_{\ell,m}(\quat{R}\_{(\vartheta,
  \varphi)})\, e^{-i\, s\, \gamma}
\end{align\*}
{:/comment}
Using orthogonality of the SWSHs, we can pull out the transformation
law for the modes:
\begin{equation}
  f^{\ell,m} = \sum\_{m'} f'^{\ell,m'}\, \mathfrak{D}^{(\ell)}\_{m',m} (\quat{R}).
\end{equation}

We should note the meaning of $\quat{R}$, now that we understand how
it fits into the transformations.  It is a rotation taking a physical
system $f$ with some value at the point $(\vartheta, \varphi)$ into a
physical system $f'$ with that value at the point $(\vartheta',
\varphi')$.



---

[^1]: We will frequently assume generality with respect to $s$ when
      making statements such as this.  In this case, $s=0$ fields are
      actually scalars, and do actually transform trivially.

[^2]: Footnote 2 of
      [the NR data-formats paper](http://arxiv.org/abs/0709.0093)
      stipulates that $\boldsymbol{m}^a = (\boldsymbol{\vartheta}^a +
      i\, \boldsymbol{\varphi}^a) / \sqrt{2}$, while Eq. (3.1) of
      [Newman-Penrose's introduction of SWSHs](http://link.aip.org/link/?JMP/7/863/1)
      states that $\boldsymbol{m}' = \boldsymbol{m}\, e^{i\, \psi}$.
      It is not hard to see that a rotation through an angle $\gamma$
      in the positive sense about the vector pointing to $(\vartheta,
      \varphi)$ gives us $\boldsymbol{m} \mapsto \boldsymbol{m}' =
      \boldsymbol{m}\, e^{-i\, \gamma}$.  Thus, if we are to use the
      standard NR conventions, we will have some negative signs
      relative to Newman and Penrose.  (Though Newman and Penrose were
      never specific enough for us to actually be in conflict with
      them; essentially $\gamma=-\psi$, which is perfectly cromulent.
      It's also reasonable to view this as a distinction between
      active and passive transformations; we assume passive in this
      case.)
