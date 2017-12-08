#!/usr/bin/env python

# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from distutils.core import setup
from auto_version import calculate_version, build_py_copy_version
from os import getenv

# Construct the version number from the date and time this python version was created.
from os import environ
if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        from subprocess import check_output
        version = check_output("""git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""", shell=use_shell).decode('ascii').rstrip()
        print("Setup.py using git log version='{0}'".format(version))
    except:
        from time import strftime, gmtime
        version = strftime("%Y.%m.%d.%H.%M.%S", gmtime())
        print("Setup.py using strftime version='{0}'".format(version))
with open('_version.py', 'w') as f:
    f.write('__version__ = "{0}"'.format(version))

validate = True
error_on_invalid = False
if getenv('CI') is not None:
    if getenv('CI').lower() == 'true':
        error_on_invalid = True

setup(name='spherical-functions',
      description='Python/numba implementation of Wigner D Matrices, spin-weighted spherical harmonics, and associated functions',
      url='https://github.com/moble/spherical_functions',
      author='Michael Boyle',
      author_email='',
      packages=['spherical_functions', ],
      package_dir={'spherical_functions': '.'},
      package_data={'spherical_functions': ['*.npy']},
      version=calculate_version(validate, error_on_invalid),
      cmdclass={'build_py': build_py_copy_version}, )
