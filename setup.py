#!/usr/bin/env python

# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from distutils.core import setup
from os import getenv

# Construct the version number from the date and time this python version was created.
from os import environ
from sys import platform
on_windows = ('win' in platform.lower() and not 'darwin' in platform.lower())
if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        # For cases where this is being installed from git.  This gives the true version number.
        from subprocess import check_output
        if on_windows:
            version = check_output("""git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""", shell=False)
            version = version.decode('ascii').strip().replace('.0', '.').replace("'", "")
        else:
            try:
                from subprocess import DEVNULL as devnull
                version = check_output("""git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S'""", shell=True, stderr=devnull)
            except AttributeError:
                from os import devnull
                version = check_output("""git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S'""", shell=True, stderr=devnull)
            version = version.decode('ascii').rstrip()
        print("Setup.py using git log version='{0}'".format(version))
    except Exception:
        # For cases where this isn't being installed from git.  This gives the wrong version number,
        # but at least it provides some information.
        #import traceback
        #print(traceback.format_exc())
        try:
            from time import strftime, gmtime
            try:
                version = strftime("%Y.%-m.%-d.%-H.%-M.%-S", gmtime())
            except ValueError:  # because Windows
                version = strftime("%Y.%m.%d.%H.%M.%S", gmtime()).replace('.0', '.')
            print("Setup.py using strftime version='{0}'".format(version))
        except:
            version = '0.0.0'
            print("Setup.py failed to determine the version; using '{0}'".format(version))
with open('spherical_functions/_version.py', 'w') as f:
    f.write('__version__ = "{0}"'.format(version))

validate = True
error_on_invalid = False
if getenv('CI') is not None:
    if getenv('CI').lower() == 'true':
        error_on_invalid = True

long_description = """\
Python/numba package for evaluating and transforming Wigner's 𝔇 matrices, Wigner's 3-j symbols, and
spin-weighted (and scalar) spherical harmonics. These functions are evaluated directly in terms of
quaternions, as well as in the more standard forms of spherical coordinates and Euler angles.
"""

if __name__ == "__main__":
    from setuptools import setup
    setup(name='spherical-functions',
        description='Python/numba implementation of Wigner D Matrices, spin-weighted spherical harmonics, and associated functions',
        long_description=long_description,
        url='https://github.com/moble/spherical_functions',
        author='Michael Boyle',
        author_email='mob22@cornell.edu',
        packages=[
            'spherical_functions',
            'spherical_functions.WignerD',
            'spherical_functions.SWSH',
            'spherical_functions.SWSH_modes',
            'spherical_functions.SWSH_grids',
            'spherical_functions.recursions',
        ],
        package_data={'spherical_functions': ['*.npy']},
        version=version,
        zip_safe=False,
        install_requires=[
            'numpy>=1.13',
            'numpy-quaternion',
        ],
        python_requires='>=3.6',
    )
