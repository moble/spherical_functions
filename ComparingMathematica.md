## Comparison with Mathematica

I find it nearly impossible to decipher the *meaning* behind Mathematica's documentation (or anyone
else's for that matter), so the only comparison I'm willing to make is between actual results when
using different codes.  Specifically, the Mathematica expression

```mathematica
WignerD[{j, m1, m2}, psi, theta, phi]
```

results in a quantity that is identical to the output from my code with this expression:

```python
(-1)**(m1+m2) * spherical_function.Wigner_D_element(quaternion.from_euler_angles(psi, theta, phi), j, m1, m2)
```

That is, if you chose actual values for `j, m1, m2, psi, theta, phi`, the numbers output by these
two expressions would be identical (within numerical precision).

Note besides the different syntax, the only real difference is the factor of $(-1)^{m_1+m_2}$.  This
is presumably related to the choice of Condon-Shortley phase.
