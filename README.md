# Simphony: A Simulator for Photonic Circuits

<div style="text-align: center">
<img alt="Development version" src="https://img.shields.io/badge/master-v0.7.1-informational">
<a href="https://pypi.python.org/pypi/simphony"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/simphony.svg"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/simphony">
<a href="https://github.com/BYUCamachoLab/simphony/actions?query=workflow%3A%22build+%28pip%29%22"><img alt="Build Status" src="https://github.com/BYUCamachoLab/simphony/workflows/build%20(pip)/badge.svg"></a>
<a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit" style="max-width:100%;"></a>
<a href="https://simphonyphotonics.readthedocs.io/"><img alt="Documentation Status" src="https://readthedocs.org/projects/simphonyphotonics/badge/?version=latest"></a>
<a href="https://pypi.python.org/pypi/simphony/"><img alt="License" src="https://img.shields.io/pypi/l/simphony.svg"></a>
<a href="https://github.com/BYUCamachoLab/simphony/commits/master"><img alt="Latest Commit" src="https://img.shields.io/github/last-commit/BYUCamachoLab/simphony.svg"></a>

<img src="https://github.com/BYUCamachoLab/simphony/blob/v0.7.1beta/docs/simphony_logo.png?raw=true" style="max-width: 500px" alt="Simphony logo">
</div>

Simphony, a simulator for photonic circuits, is a fundamental package for designing and simulating photonic integrated circuits with Python.

**Key Features:**

- Free and open-source software provided under the MIT License
- Completely scriptable using Python 3.
- Cross-platform: runs on Windows, MacOS, and Linux.
- Subnetwork growth routines
- A simple, extensible framework for defining photonic component compact models.
- A SPICE-like method for defining photonic circuits.
- Complex simulation capabilities.
- Included model libraries from SiEPIC and SiPANN.

Developed by [CamachoLab](https://camacholab.byu.edu/) at
[Brigham Young University](https://www.byu.edu/).

## Installation

Simphony can be installed via pip for Python 3.8+:

```
python3 -m pip install simphony
```

## Documentation

The documentation is hosted [online](https://simphonyphotonics.readthedocs.io/en/latest/).

## Bibtex citation

```
@article{DBLP:journals/corr/abs-2009-05146,
  author    = {Sequoia Ploeg and
               Hyrum Gunther and
               Ryan M. Camacho},
  title     = {Simphony: An open-source photonic integrated circuit simulation framework},
  journal   = {CoRR},
  volume    = {abs/2009.05146},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.05146},
  eprinttype = {arXiv},
  eprint    = {2009.05146},
  timestamp = {Thu, 17 Sep 2020 12:49:52 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2009-05146.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Development

We welcome contributions to Simphony in the form of bug reports, feature
requests, and pull requests. Please see the [contributing
guide](https://simphonyphotonics.readthedocs.io/en/stable/dev/contributing.html).

To develop locally, clone the repository and run the ``make`` commands to set
up your development environment.

```bash
git clone https://github.com/BYUCamachoLab/simphony.git
python3 -m venv env   # or use your preferred virtual environment tool
source env/bin/activate
make install
```

We use pre-commit to maintain code quality. The hooks are automatically 
installed when invoking the make targets. Pre-commit will now run on every
commit. If it makes any modifications, the commit will fail and you'll have
to restage the changes before continuing with the commit.

If you truly, desperately need to skip pre-commit in one instance, you can use:

```bash
git commit --no-verify -m "Commit message"
```

Pleasee don't make a habit of it.

There are a few other useful targets in the Makefile:

- ``make test``: Run the unit tests
- ``make doc``: Build the documentation

## Distribution

Simphony is available on PyPI and can be installed via pip:

```
python3 -m pip install simphony
```

## License

Simphony is licensed under the MIT license. See the [LICENSE](LICENSE) file for
more details.

## Releasing (Danger Zone)

**Only the project maintainer should create releases.**

A note on the development cycle. The ``master`` branch is the "latest" branch
with a version of the project that *always works*. Features and bug fixes are
developed in separate branches and then merged into master when ready.

When preparing for a new release, it's appropriate to increment the version
number in advance of the release and before all the changes are merged in. You 
can start the next version and increment the version number by running:

```
make major
make minor
make patch
```

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
which means that the version number is incremented according to the following
rules:

* MAJOR version when you make incompatible API changes,
* MINOR version when you add functionality in a backwards compatible manner, and
* PATCH version when you make backwards compatible bug fixes.

There are also software lifecycle parts to the version number:

* BUILD version is for multiple versions within a release (i.e. "rc1", "rc2", etc.)
* RELEASE version cycles through "rc(n)" and then empty string for final release.

A version with a "rc" number indicates it is not a final release. When you're
ready to create the final release, make sure you've filled out the
corresponding entry in the [CHANGELOG](CHANGELOG.md). Follow the format of
previous recent entries. Make sure you have a clean working tree (git). Then,
run:

```
bumpversion release
make release
```

This will remove the "rc" number, tag the commit, push the tag to git, upload
a build to PyPI, and publish the documentation to ReadTheDocs. Because this
process is irreversible and version numbers cannot be reclaimed from PyPI, make
sure all tests are passing and the documentation is up to date before running
this command.

A standard version number progression might look something like this:

* 2.0.1
* 2.0.2rc0
* 2.0.2rc1
* 2.0.2
