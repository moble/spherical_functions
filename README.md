[![Test Status](https://github.com/moble/spherical_functions/workflows/tests/badge.svg)](https://github.com/moble/spherical_functions/actions)
[![PyPI Version](https://img.shields.io/pypi/v/spherical-functions?color=)](https://pypi.org/project/spherical-functions/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/spherical_functions.svg?color=)](https://anaconda.org/conda-forge/spherical_functions)
[![MIT License](https://img.shields.io/github/license/moble/spherical_functions.svg)](https://github.com/moble/spherical_functions/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/25589782.svg)](https://zenodo.org/badge/latestdoi/25589782)


# Spherical Functions

| NOTE: This package will still be maintained, but *active* development has moved to the [`spherical`](https://github.com/moble/spherical) package.  While this package works well for ℓ (aka ell, L, j, or J) values up to around 25, errors start to build rapidly and turn into NaNs around 30.  The `spherical` package can readily handle values up to at least 1000, with accuracy close to ℓ times machine precision.  —Mike |
| --- |


Python/numba package for evaluating and transforming Wigner's 𝔇 matrices,
Wigner's 3-j symbols, and spin-weighted (and scalar) spherical harmonics.
These functions are evaluated directly in terms of quaternions, as well as in
the more standard forms of spherical coordinates and Euler
angles.<sup>[1](#1-euler-angles-are-awful)</sup>

The conventions for this package are described in detail on
[this page](http://moble.github.io/spherical_functions/).

## Dependencies

The only true requirements for this code are `python` and the python package
`numpy`, as well as my accompanying
[`quaternion`](https://github.com/moble/quaternion) package (installation of
which is shown below).

However, this package can automatically use
[`numba`](http://numba.pydata.org/), which uses [LLVM](http://llvm.org/) to
compile python code to machine code, accelerating most numerical functions by
factors of anywhere from 2 to 2000.  It is *possible* to run the code without
`numba`, but the most important functions are roughly 10 times slower without
it.

The only drawback of `numba` is that it is nontrivial to install on its own.
Fortunately, the best python installer,
[`anaconda`](http://continuum.io/downloads), makes it trivial.  Just install
the main `anaconda` package.

If you prefer the smaller download size of
[`miniconda`](http://conda.pydata.org/miniconda.html) (which comes with no
extras beyond python), you may also have to run this command:

```sh
conda install pip numpy numba
```


## Installation

Assuming you use `conda` to manage your python installation (like any sane
python user), you can install this package simply as

```sh
conda install -c conda-forge spherical_functions
```

This should automatically download and install the package
[`quaternion`](https://github.com/moble/quaternion), on which this package
depends.

Alternatively, if you prefer to use `pip` (whether or not you use `conda`), you
can also do

```sh
python -m pip install spherical-functions
```

Finally, there's also the fully manual option of just downloading both code
repositories, changing to the code directory, and issuing

```sh
python -m pip install .
```

This should work regardless of the installation method, as long as you have a
compiler hanging around.  However, this may be more likely to try to compile
the dependencies, including numpy and/or spinsfast, which can be much more
complicated.


## Usage

First, we show a very simple example of usage with Euler angles, though it
breaks my heart to do so:<sup>[1](#euler-angles-are-awful)</sup>

```python
>>> import spherical_functions as sf
>>> alpha, beta, gamma = 0.1, 0.2, 0.3
>>> ell,mp,m = 3,2,1
>>> sf.Wigner_D_element(alpha, beta, gamma, ell, mp, m)

```

Of course, it's always better to use unit quaternions to describe rotations:

```python
>>> import numpy as np
>>> import quaternion
>>> R = np.quaternion(1,2,3,4).normalized()
>>> ell,mp,m = 3,2,1
>>> sf.Wigner_D_element(R, ell, mp, m)

```

If you need to calculate values of the 𝔇<sup>(ℓ)</sup> matrix elements for many
values of (ℓ, m', m), it is more efficient to do so all at once.  The following
calculates all modes for ℓ from 2 to 8 (inclusive):

```python
>>> indices = np.array([[ell,mp,m] for ell in range(2,9)
... for mp in range(-ell, ell+1) for m in range(-ell, ell+1)])
>>> sf.Wigner_D_element(R, indices)

```

Finally, if you really need to put the pedal to the metal, and are willing to
guarantee that the input arguments are correct, you can use a special hidden
form of the function:

```python
>>> sf._Wigner_D_element(R.a, R.b, indices, elements)

```

Here, `R.a` and `R.b` are the two complex parts of the quaternion defined on
[this page](http://moble.github.io/spherical_functions/) (though the user need
not care about that).  The `indices` variable is assumed to be a
two-dimensional array of integers, where the second dimension has size three,
representing the (ℓ, m', m) indices.  This avoids certain somewhat slower
pure-python operations involving argument checking, reshaping, etc.  The
`elements` variable must be a one-dimensional array of complex numbers (can be
uninitialized), which will be replaced with the corresponding values on return.
Again, however, there is no input dimension checking here, so if you give bad
inputs, behavior could range from silently wrong to exceptions to segmentation
faults.  Caveat emptor.


## Acknowledgments

I very much appreciate Barry Wardell's help in sorting out the relationships
between my conventions and those of other people and software packages
(especially Mathematica's crazy conventions).

This code is, of course, hosted on github.  Because it is an open-source
project, the hosting is free, and all the wonderful features of github are
available, including free wiki space and web page hosting, pull requests, a
nice interface to the git logs, etc.

Finally, the code is automatically compiled, and the binaries hosted for
download by `conda` on [anaconda.org](https://anaconda.org/moble/spherical_functions).
This is also a free service for open-source projects like this one.

The work of creating this code was supported in part by the Sherman Fairchild
Foundation and by NSF Grants No. PHY-1306125 and AST-1333129.


<br/>
---
###### <sup>1</sup> Euler angles are awful

Euler angles are pretty much
[the worst things ever](http://moble.github.io/spherical_functions/#euler-angles)
and it makes me feel bad even supporting them.  Quaternions are
faster, more accurate, basically free of singularities, more
intuitive, and generally easier to understand.  You can work entirely
without Euler angles (I certainly do).  You absolutely never need
them.  But if you're so old fashioned that you really can't give them
up, they are fully supported.
