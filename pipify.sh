set -e

/bin/rm -rf build dist sxs.egg-info
pip install --quiet --upgrade twine
python setup.py sdist bdist_wheel --universal
twine upload dist/*
