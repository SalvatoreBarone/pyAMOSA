#!/bin/bash
source venv/bin/activate
rm -rf build dist pyAMOSA.egg-info
python setup.py check
python setup.py sdist bdist_wheel
python -m twine upload dist/*