---
---

# Polynomials with factorials

We know that Horner form is a very efficient way to evaluate polynomials,
but if the coefficients of that polynomial involve factorials of the index
itself, we can incorporate a further simplification.  The standard Horner
form of a general polynomial is
\begin{equation\*}
  \sum\_{\rho=\rho\_0} c\_\rho x^\rho = x^{\rho\_0} \left( c\_{\rho\_0} + x
  \left(c\_{\rho\_0+1} + x \left( c\_{\rho\_0+2} + \ldots \right) \right)
  \right).
\end{equation\*}
If `c` is a function that returns the appropriate coefficient, we can
implement this with code like

```python
tot = 0.0
for i in range(rho_max, rho_min-1, -1):
    tot = c(i) + x * tot
tot *= x**(rho_min)
```

Note that `c(i)` is always *added to* the current total in this algorithm.

Now, if instead, the coefficients are all $\rho!$, we have
\begin{equation\*}
  \sum\_{\rho=\rho\_0} \rho! x^\rho = (\rho\_0)! x^{\rho\_0} \left( 1 +
  (\rho\_0+1) x \left(1 + (\rho\_0+2) x \left( 1 + \ldots \right) \right)
  \right).
\end{equation\*}
The efficient code for this is

```python
tot = 1.0
for rho in range(rho_max, rho_min, -1):
    tot = 1 + rho * x * tot
tot *= factorial(rho_min)*x**rho_min
```

Here, the coefficient *multiplies* the current total.

Now, a more interesting case for our purposes involves a more complicated
coefficient:
\begin{equation\*}
  \sum\_{\rho=\rho\_0} \frac{(M+\rho\_0)!} {(M+\rho)!} x^\rho = x^{\rho\_0}
  \left( 1 + \frac{1} {M+\rho\_0+1} x \left(1 + \frac{1} {M+\rho\_0+2} x
  \left( 1 + \ldots \right) \right) \right).
\end{equation\*}
Here again, we can turn this into a simple algorithm:

```python
tot = 1.0
for rho in range(rho_max, rho_min, -1):
    tot = 1 + x * tot / (M+rho)
tot *= x**rho_min
```

Another interesting case for our purposes involves $-\rho$:
\begin{equation\*}
  \sum\_{\rho=\rho\_0} \frac{(N-\rho\_0)!} {(N-\rho)!} x^\rho = x^{\rho\_0}
  \left( 1 + (N-\rho\_0) x \left(1 + (N-\rho\_0-1) x
  \left( 1 + \ldots \right) \right) \right).
\end{equation\*}
Here again, we can turn this into a simple algorithm:

```python
tot = 1.0
for rho in range(rho_max, rho_min, -1):
    tot = 1 + x * tot * (N-rho+1)
tot *= x**rho_min
```

And, of course, we can combine them.

