===================
Releasing a Version
===================


For Developers
--------------

This package is available on PyPI and updates are regularly pushed as "minor" 
or "micro" (patch) versions. Before submitting any pull requests, however, you should 
ensure that a pip installation of your updated package installs and functions 
properly. To test this, try installing your package locally by removing all 
installed versions of Simphony (by running ```pip3 uninstall simphony``` 
repeatedly until no installations remain) and running the following commands 
(from Simphony's toplevel directory):

```
python3 setup.py sdist bdist_wheel
pip3 install dist/simphony-[VERSION].tar.gz
```

For Maintainers
---------------

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

Eventually, as the project grows, we will work up to using the methods 
detailed below, retained for future reference.

------------------------
How to Prepare a Release
------------------------

.. include:: howto_release.rst

-----------------------
Step-by-Step Directions
-----------------------

.. include:: release_walkthrough.rst

