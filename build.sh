#! /bin/bash

rm -rf dist/*
python3 -m build
pip uninstall PARCtorch
#python3 -m twine upload -r testpypi dist/*
