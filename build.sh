#! /bin/bash
python3 -m build
python3 -m twine upload -r testpypi dist/*
