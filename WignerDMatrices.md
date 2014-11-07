## Wigner $\mathfrak{D}$ matrices

The Wigner $\mathfrak{D}$ matrices are derived in terms of a
particular split of the quaternion group into two parts.  Explicitly,
a quaternion $\quat{Q}$ can be expressed in terms of two complex
numbers $\quat{Q}\_a = \quat{Q}\_1 + i\, \quat{Q}\_z$ and $\quat{Q}\_b
= \quat{Q}\_y + i\, \quat{Q}\_x$.[^1]  This is only important because
it allows us to verify the multiplication law
\begin{align}
  \label{eq:QuaternionComponentProducts}
  (\quat{P}\,\quat{Q})\_a &= \quat{P}\_a\, \quat{Q}\_a - \co{\quat{P}}\_b\, \quat{Q}\_b, \\\\
  (\quat{P}\,\quat{Q})\_b &= \quat{P}\_b\, \quat{Q}\_a + \co{\quat{P}}\_a\, \quat{Q}\_b.
\end{align}
Given a rotor $\rotor{R}$, these two complex numbers are the
quantities actually used in computing the $\mathfrak{D}$ matrix
elements.

This is essentially the same as Wigner's original derivation, but with
more elegance, and more sensitivity to numerical issues and special
cases.  This version of the derivation comes from
[a paper](http://arxiv.org/abs/1302.2919) I wrote a couple years ago,
and is the source of the code used in this module.

The basic idea of the derivation is to construct a
$(2\ell+1)$-dimensional vector space of homogeneous polynomials in
these complex numbers $\quat{Q}\_a$ and $\quat{Q}\_b$.  To make that a
little more concrete, the basis of this vector space is
\begin{equation\*}
  \label{WignerBasisComponent}
  \mathbf{e}\_{(m)}(\quat{Q}) \defined
    \frac{\quat{Q}\_{a}^{\ell+m}\, \quat{Q}\_{b}^{\ell-m}}
    {\sqrt{ (\ell+m)!\, (\ell-m)! }}.
\end{equation\*}
Here, $m$ ranges from $-\ell$ to $\ell$ in steps of $1$, but $\ell$
(and hence $m$) can be a half-integer.  Now, the key idea is that a
rotation of $\quat{Q}$ by some new rotor $\quat{R}$ gives us a new
vector basis, which we can represent in terms of the basis shown
above.  In fact, we get a matrix transforming one set of basis vectors
to another.  We'll write this as
\begin{equation\*}
  \mathbf{e}\_{(m')}(\quat{R}\, \quat{Q})
  = \sum\_{m} \mathfrak{D}^{(\ell)}\_{m',m}(\quat{R})\, \mathbf{e}\_{(m)}(\quat{Q}).
\end{equation\*}

So now, we've defined the $\mathfrak{D}$ matrices.  But we can also
plug $\quat{R}\, \quat{Q}$ into the original expression for
$\mathbf{e}$, and figure out what $\mathfrak{D}$ should actually be.
We'll have to use the expressions for $(\quat{R}\, \quat{Q})_a$ and
$(\quat{R}\, \quat{Q})_b$ given above, and we'll find that we have
polynomials with terms given as sums of two different things.

This brings us to the first fork in the road.  If either $\quat{R}_a$
or $\quat{R}_b$ is tiny (i.e., the absolute value is numerically
small), we can (and in fact must) treat these as special cases.
First, if $\lvert \quat{R}_a \rvert \lesssim 10^{-14}$, for example,
we can just ignore it; since $\lvert \quat{R} \rvert=1$ (within
numerical precision), we are assured that $\lvert \quat{R}_b \rvert
\approx 1$.  Thus, we get
\begin{align\*}
  \mathbf{e}\_{(m')}(\quat{R}\, \quat{Q})
  &\approx \frac{ (- \co{\quat{R}}\_{b}\, \quat{Q}\_{b})^{\ell+m'}\,
    (\quat{R}\_{b}\, \quat{Q}\_{a})^{\ell-m'} } { \sqrt{
      (\ell+m')!\, (\ell-m')! } }, \\\\
  &\approx (- \co{\quat{R}}\_{b})^{\ell+m'}\,
  (\quat{R}\_{b})^{\ell-m'}\, \mathbf{e}\_{(-m')}(\quat{Q}).
\end{align\*}
In this case, it's not hard to see that the expression for the
$\mathfrak{D}$ matrix is
\begin{equation}
  \label{eq:D\_RaApprox0}
  \mathfrak{D}^{(\ell)}\_{m',m}(\quat{R})
  = (-1)^{\ell+m'}\, \quat{R}\_b^{-2m'} \delta\_{-m',m}
  = (-1)^{\ell+m}\, \quat{R}\_b^{2m} \delta\_{-m',m}.
\end{equation}
In the same way, we can calculate this for $\lvert \quat{R}_b \rvert
\lesssim 10^{-14}$:
{::comment}
\begin{align\*}
  \mathbf{e}\_{(m')}(\quat{R}\, \quat{Q})
  &\approx \frac{ (\quat{R}\_{a}\, \quat{Q}\_{a})^{\ell+m'}\,
    (\co{\quat{R}}\_{a}\, \quat{Q}\_{b})^{\ell-m'} } { \sqrt{
      (\ell+m')!\, (\ell-m')! } }, \\\\
  &\approx (\quat{R}\_{a})^{\ell+m'}\,
  (\co{\quat{R}}\_{a})^{\ell-m'}\, \mathbf{e}\_{(m')}(\quat{Q}).
\end{align\*}
{:/comment}
\begin{equation}
  \label{eq:D\_RbApprox0}
  \mathfrak{D}^{(\ell)}\_{m',m}(\quat{R})
  = \quat{R}\_a^{2m'} \delta\_{m',m}
  = \quat{R}\_a^{2m} \delta\_{m',m}.
\end{equation}

Now, the other fork in that road is the general case, when both
components have significant magnitudes.  We have powers of the sum of
those terms in Eq. \eqref{eq:QuaternionComponentProducts}.  This leads
us to use the
[binomial expansion](https://en.wikipedia.org/wiki/Binomial_theorem).
After a little simplification, we can express the result as
\begin{align\*}
  \mathfrak{D}^{(\ell)}\_{m',m}(\quat{R})
  = \sum\_{\rho} \binom{\ell+m'} {\rho}\, \binom{\ell-m'}
  {\ell-\rho-m}\, (-1)^{\rho}\, \quat{R}\_{a}^{\ell+m'-\rho}\,
  \co{\quat{R}}\_{a}^{\ell-\rho-m}\,
  \quat{R}\_{b}^{\rho-m'+m}\, \co{\quat{R}}\_{b}^{\rho}\,
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\,
      (\ell-m')! } }.
\end{align\*}
Now, this expression is not the best way to implement the calculation
in the code.  The reason is that we would need to take a bunch of
exponents of complex numbers, and there's the possibility that the sum
would cancel out to give a very small number, which would be polluted
with roundoff, etc.  So we try to pull out as many constants as we can
from the sum, and then try to write the sum in
[Horner form](http://reference.wolfram.com/language/ref/HornerForm.html),
which is a fast and accurate way to evaluate sums.

For example, we can simplify the above as
\begin{align}
  \nonumber
  \mathfrak{D}^{(\ell)}\_{m,m'}(\quat{R})
  &=
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\, (\ell-m')! } }\,
  \quat{R}\_{a}^{\ell+m'}\,
  \co{\quat{R}}\_{a}^{\ell-m}\,
  \quat{R}\_{b}^{-m'+m} \\\\
  \nonumber
  &\qquad
  \times \sum\_{\rho}
  \binom{\ell+m'} {\rho}\, \binom{\ell-m'} {\ell-\rho-m}\,
  (-1)^{\rho}\,
  \quat{R}\_{a}^{-\rho}\,
  \co{\quat{R}}\_{a}^{-\rho}\,
  \quat{R}\_{b}^{\rho}\,
  \co{\quat{R}}\_{b}^{\rho}, \\\\
  \nonumber
  &=
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\, (\ell-m')! } }\,
  \lvert \quat{R}\_{a} \rvert^{2\ell-2m}\,
  \quat{R}\_{a}^{m'+m}\,
  \quat{R}\_{b}^{-m'+m} \\\\
  \label{eq:D\_RaGeqRb}
  &\qquad
  \times \sum\_{\rho}
  \binom{\ell+m'} {\rho}\, \binom{\ell-m'} {\ell-\rho-m}\,
  (-1)^{\rho}\, \left( \frac{\lvert \quat{R}\_{b} \rvert^2}
  {\lvert \quat{R}\_{a} \rvert^2} \right)^{\rho}.
\end{align}
This last expression can be evaluated pretty efficiently.  We just
have two complex exponentials and one real exponential out front.  The
coefficients (the square root and the binomials) can be calculated
ahead of time and cached, and the ratio in the sum can be evaluated
once, then reused.  The exponentiation of this is done implicitly
using Horner form.

There are two things to be careful of when applying this formula.
First, the sum is well behaved when the ratio at the end is small, so
let's say when $\lvert \quat{R}\_b \rvert < \lvert \quat{R}\_a
\rvert$.  However, if this is the case and $\lvert \quat{R}\_b \rvert$
is *too* small, we might actually run into problems with underflow, in
which case the exponential will come out as zero.  This can happen,
for example, if we need to compute an $\ell=16$ matrix when $\lvert
\quat{R}\_b \rvert \lesssim 10^{-11}$, and $m'=-16$ and $m=16$, in
which case $\lvert \quat{R}\_b \rvert^{-m'+m} \lesssim 10^{-352}$,
which is less than the smallest `float` that python can represent, so
it goes to zero.

So we need to combine knowledge of the magnitude of $\lvert
\quat{R}\_b \rvert$ with knowledge of the power to which it will be
raised.

So we get four branches in our logic, with a different expression for
$\mathfrak{D}$ in each branch:

  1. When $\lvert \quat{R}\_a \rvert \lesssim 10^{-14}$, use
     Eq. \eqref{eq:D\_RaApprox0}.
  2. When $\lvert \quat{R}\_b \rvert \lesssim 10^{-14}$, use
     Eq. \eqref{eq:D\_RbApprox0}.
  3. When $\lvert \quat{R}\_a \rvert \geq \lvert \quat{R}\_b \rvert$,
     use Eq. \eqref{eq:D\_RaGeqRb}.
  4. When $\lvert \quat{R}\_a \rvert < \lvert \quat{R}\_b \rvert$, use
     Eq. {}.


[^1]: This, of course, is not a productive way of *thinking about*
      quaternions, but it is a very useful way of *calculating with*
      quaternions, since complex numbers are already built in to many
      languages.  That is, this decomposition into two complex
      components is something that the user probably does not need to
      worry about.  It is, however, an isomorphism between quaternions
      and the usual (awful) presentation of Pauli spinors as
      two-component complex vectors, which is closer to the flavor of
      Wigner's original derivation.  It is also how the code is
      actually implemented.

