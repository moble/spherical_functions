---
---

# Comparing conventions

## Goldstein-Poole-Safko "Classical Mechanics"

### Euler angles

 1. $\phi$ about $z$
 2. $\theta$ about $x'$
 3. $\psi$ about $z''$

Goldstein (p. 151) claims that this convention is widely used in
celestial mechanics and applied mechanics, and frequently in molecular
and solid-state physics.  Goldstein also has an Appendix on the
various conventions.  I would write Goldstein's rotation as
\begin{equation}
  \label{eq:GoldsteinEulerAngles}
  e^{\phi \basis{z}/2}\, e^{\theta \basis{x}/2}\, e^{\psi \basis{z}/2}.
\end{equation}
That's right, he uses $x$!  I think this means that my coordinate
$\gamma$ would be related to his $\psi$ according to $\gamma = \psi -
\pi/2$.

### Wigner $\mathfrak{D}$; harmonics

I know of nowhere that Goldstein uses either


## Shankar


## Sakurai


## Wikipedia


## Mathematica

The Euler angles correspond to what I would have considered the
*inverse* rotation.

## Sympy

[Sympy](http://sympy.org/en/index.html) (the symbolic math package for
python) implements the $\mathfrak{D}$ matrices as the function
`sympy.physics.quantum.spin.WignerD`.  Note that
[the documentation](http://docs.sympy.org/latest/modules/physics/quantum/spin.html#sympy.physics.quantum.spin.WignerD)
swaps the symbols $m$ and $m'$ in their documentation, relative to the
usual order.  Nonetheless, the arguments to the function are in the
standard order $(m',m)$.  That is, if you call the function using what
the rest of this module (and other items on this page) regard as $m'$
and $m$, you will get the expected result---up to the interpretation
of the Euler angles.  Don't be alarmed by the fact that the
documentation lists the arguments as $(m,m')$.


## Wigner

(As translated by J. J. Griffin)

The $\mathfrak{D}$ matrix is given on page 167, Eq. (15.27).
