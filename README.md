# spherical_functions


Python/numba package for evaluating and transforming Wigner 𝔇 matrices
and spin-weighted spherical harmonics directly in terms of
quaternions, as well as in the more standard forms of spherical
coordinates and Euler angles.

The conventions for this package are described in detail on
[this page](http://moble.github.io/spherical_functions/).

## Dependencies

The only true requirements for this code are `python` and `numpy`.

However, this package can automatically use
[`numba`](http://numba.pydata.org/), which uses
[LLVM](http://llvm.org/) to compile python code to machine code,
accelerating most numerical functions by factors of anywhere from 2
to 2000.  It is *possible* to run the code without `numba`, but the
most important functions are roughly 10 times slower without it.

The only drawback of `numba` is that it is nontrivial to install on
its own.  Fortunately, the best python installer,
[`anaconda`](http://continuum.io/downloads), makes it trivial.  Just
install the main `anaconda` package.

If you prefer the smaller download size of
[`miniconda`](http://conda.pydata.org/miniconda.html) (which comes
with no extras beyond python), you'll also have to run this command:

```sh
conda install pip numpy numba
```


## Installation

Installation of this package is simple:

```sh
pip install git+git://github.com/moble/spherical_functions@master
```


## Usage

