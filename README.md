# Simphony
A Simulator for Photonic circuits

![Text](./docs/source/_static/images/simphony_logo.jpg)

Authors: [Sequoia Ploeg](https://github.com/sequoiap), 
[Hyrum Gunther](https://github.com/rumbonium/)

Developed by [CamachoLab](https://camacholab.byu.edu/) at 
[Brigham Young University](https://www.byu.edu/).

# About this branch

Previous development branches required the component models (from which instances
are created) to be instantiated first. This attempt tries to keep them as simple
classes, removing the requirement to instantiate. It also tries to keep the
s-parameters with the classes, without so many file i/o and parsing algorithms.

# Description

This package is still under development. It initially began as an extension to
[SiEPIC-Tools](https://github.com/lukasc-ubc/SiEPIC-Tools), but was ported here
once it became large enough to be considered its own stand-alone project. There
is a repository forked from lukasc-ubc/SiEPIC-Tools, 
[SiEPIC-Tools](https://github.com/sequoiap/SiEPIC-Tools),
that integrates Simphony with SiEPIC-Tools and KLayout in order to perform 
photonic circuit simulations using a layout-driven design methodology.

Simphony can be installed via pip using Python 3:

```
pip install simphony
```

Please note that Python 2 is not supported. With the looming deprecation of
Python 2 (January 1, 2020), no future compatability is planned.

## Documentation

Documentation is built on Sphinx. They can be built using the default files by 
navigating to the docs directory and running:

```
make html
```

The docs are written in reST. There is a nice syntax guide with guidelines that
we follow in the documentation 
[here](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html). The 
majority of the documentation is generated from python docstrings written using
the NumPy documentation format.

Changelogs can be found in docs/changelog/. There is a changelog file for 
each released version of the software.

## Tests

Simphony uses the [pytest](https://docs.pytest.org/en/latest/) testing 
framework. To run all tests, simply execute

```
pytest
```

from the toplevel directory.

## Developers

This package is available on PyPI and updates are regularly pushed as "minor" 
or "micro" (patch) versions. Before submitting any pull requests, however, you should 
ensure that a pip installation of your updated package installs and functions 
properly. To test this, try installing your package locally by removing all 
installed versions of Simphony (by running ```pip uninstall simphony``` 
repeatedly until no installations remain) and running the following commands 
(from Simphony's toplevel directory):

```
python3 setup.py sdist bdist_wheel
pip install dist/simphony-[VERSION].tar.gz
```
## Contributing

All contributions and new features or bug fixes should be worked on in forks
or branches of the repository. Issues should be opened, and pull requests
should reference and [close those issues](https://help.github.com/en/articles/closing-issues-using-keywords).
This is good versioning and documentation practice.

## Maintainers

Remember that all changes are to be integrated through pull requests. Development
work should be done in branches or forks of the repository. Once implemented 
(and tested on their own), these pull requests should be merged into the 
"master" branch for full testing with the whole program. Each time the package
is released on PyPI, the package should have a pull request opened to its 
corresponding release branch (release-MAJOR.MINOR.x). The hierarchy is then
as follows:

- release-*.*.x (stable branch)
- master (integration and final testing)
- feature-name (feature development and bug fixes)

Even if you are the lone developer, we follow the methodology [here](https://softwareengineering.stackexchange.com/a/294048).

Be sure to update the version number manually before pushing each new version 
to PyPI. Also be sure to amend the changelog. Versions can be pushed to PyPI 
using the commands:

```
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
```