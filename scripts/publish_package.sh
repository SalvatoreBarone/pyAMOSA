#!/bin/bash
source venv/bin/activate
rm -rf build dist pyAMOSA.egg-info
python3 setup.py check
python3 setup.py sdist
python3 -m twine upload dist/*