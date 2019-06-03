# Simphony
Simulator for Photonic circuits

This package is still under development. It began as an extension to SiEPIC-Tools, but became large enough to become its own stand-alone project. The first minor version is considered what was under develop during its time with SiEPIC-Tools.

Simphony can be installed via pip:

pip install simphony


## Developers

This package is available on PyPI and updates are regularly pushed as "minor" or "micro" versions. Before submitting any pull requests, however, you should ensure that a pip installation of your updated package installs and functions properly. To test this, try installing your package locally by removing all installed versions of Simphony (by running ```pip uninstall simphony``` repeatedly until no installations remain) and running the following commands (from Simphony's toplevel directory):

```
python3 setup.py sdist bdist_wheel
pip install dist/simphony-[VERSION].tar.gz
```

## Maintainers

Be sure to update the version number manually before pushing each new version to PyPI. Also be sure to amend the changelog. Versions can be pushed to PyPI using the command:

```
python3 -m twine upload dist/*
```