Python/numba package for evaluating and transforming Wigner's 𝔇 matrices, Wigner's 3-j symbols, and spin-weighted
(and scalar) spherical harmonics. These functions are evaluated directly in terms of quaternions, as well as in the
more standard forms of spherical coordinates and Euler angles.(*)

    https://github.com/moble/spherical_functions/blob/master/README.md



(*) Euler angles are pretty much the worst things ever, and it makes me feel bad even supporting them.  Quaternions
are faster, more accurate, basically free of singularities, more intuitive, and generally easier to understand.  You
can work entirely without Euler angles (I certainly do).  You absolutely never need them.  But if you're so old
fashioned that you really can't give them up, they are fully supported.
