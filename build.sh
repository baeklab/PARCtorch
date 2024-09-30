#! /bin/bash
python3 -m build
pip uninstall PARCtorch
#python3 -m twine upload -r testpypi dist/*
