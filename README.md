# Simphony
Description: A Simulator for Photonic circuits

Authors: [Sequoia Ploeg](https://github.com/sequoiap), [Hyrum Gunther](https://github.com/rumbonium/)

Developed by [CamachoLab](https://camacholab.byu.edu/) at [Brigham Young University](https://www.byu.edu/).

This package is still under development. It initially began as an extension to [SiEPIC-Tools](https://github.com/lukasc-ubc/SiEPIC-Tools), but was ported here once it became large enough to be considered its own stand-alone project. There is a repository, [SiEPIC-Simphony](https://github.com/sequoiap/SiEPIC-Simphony), that integrates Simphony with SiEPIC-Tools and KLayout in order to perform photonic circuit simulations using a layout-driven design methodology.

Simphony can be installed via pip:

```
pip install simphony
```

## Documentation

Documentation is built on Sphinx. They can be built using the default files by navigation to the docs directory and running:

```
make html
```

The docs are written in reST. There is a nice syntax guide with guidelines that we follow in the documentation [here](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html).

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