#! /bin/bash

set -e

/bin/rm -rf build __pycache__ dist spherical_functions.egg-info
python setup.py install
python -c 'import spherical_functions as sf; print(sf.__version__)'

pip install --quiet --upgrade twine
/bin/rm -rf dist
python setup.py sdist
twine upload dist/*
