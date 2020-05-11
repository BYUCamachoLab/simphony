This file gives an overview of what is necessary to build binary releases for
Simphony.


Supported platforms and versions
================================

Platform Independent
--------------------

As a relatively straightforward Python-only package, Simphony doesn't require
any special builds for different operating systems and should be universally 
installable via pip.


Tool chain
==========

We build all our wheels locally on Linux.

.. on cloud infrastructure - so this list of compilers is
.. for information and debugging builds locally.  See the ``.travis.yml`` and
.. ``appveyor.yml`` scripts in the `numpy wheels`_ repo for the definitive source
.. of the build recipes. Packages that are available using pip are noted.


Building source archives and wheels
-----------------------------------

- Python(s) from `python.org <https://python.org>`_ or linux distro.
- virtualenv (pip)


Building docs
-------------
Building the documents requires the items listed below and reproduced in a
doc_requirements.txt file.

- Sphinx (pip)


Uploading to PyPI
-----------------
- twine (pip)


What is released
================

Wheels
------
.. We currently support Python 3.6-3.8 on Windows, OSX, and Linux

.. * Windows: 32-bit and 64-bit wheels built using Appveyor;
.. * OSX: x64_86 OSX wheels built using travis-ci;
.. * Linux: 32-bit and 64-bit Manylinux1 wheels built using travis-ci.

Simphony ought to be OS-independent. Hence, a ``none-any`` built wheel is 
included in the release.


Other
-----
- Changelog


Source distribution
-------------------
We build source releases in both .zip and .tar.gz formats.


Release process
===============


Make sure current branch builds a package correctly
---------------------------------------------------
::

    git clean -fxd
    python setup.py bdist
    python setup.py sdist

.. note:: The following steps are repeated for the beta(s), release
   candidates(s) and the final release.


Check deprecations
------------------
Before the release branch is made, it should be checked that all deprecated
code that should be removed is actually removed, and all new deprecations say
in the docstring or deprecation warning at what version the code will be
removed.


Check the changelog
-------------------

Check that the changelog, which is handwritten, is up-to-date.

After the first paragraph summary/overview, perhaps mention some of the
following highlights:

  - major new features
  - deprecated and removed features
  - supported Python versions
  - outlook for the near future

A template for a typical changelog is as follows: ::

    ## [MAJOR.MINOR.PATCH] - YEAR-MM-D

    This section provides an overview/summary of the release.

    Any other highlights, as necessary.

    ### Added
    - [#<val>](<link-to-issue>) Description of what was added.

    ### Changed
    - List of changes.

    ### Removed
    - List of items removed, if any.


Update the release status and create a release "tag"
----------------------------------------------------

Make sure all the code to be included in the release is up to date and pushed 
to ``master``. 

Go to `<https://github.com/BYUCamachoLab/simphony/releases>`_, the main
Simphony repository in GitHub, and draft a new release. The name of the release
should be the version number, in the following format: ::

    v<MAJOR>.<MINOR>.<PATCH> # e.g. v0.3.0


Update the version of the master branch
---------------------------------------
Increment the release number in ``simphony/__init__.py``. 
Release candidates should have "rc1"
(or "rc2", "rcN") appended to the X.Y.Z format.


Trigger the wheel builds
------------------------

We use ``setuptools`` and ``wheel`` to package Simphony. Make sure you have
the latest version installed: ::

    python3 -m pip install --user --upgrade setuptools wheel

In the same directory as ``setup.py``, run the following command: ::

    python3 setup.py sdist --formats=gztar,zip bdist_wheel

It will create the ``dist`` directory and place within it the ``*.zip`` and ``*.tar.gz``
source releases, as well as the built distribution ``*.whl``. Since Simphony
is not OS-specific (at least for now), the single wheel should be good for 
any platform.


Build and archive documentation
-------------------------------
Do::

    cd doc/
    make dist

to check that the documentation is in a buildable state. Then, after tagging
a release in GitHub, activate the documentation version online at the web 
interface at `ReadTheDocs <https://readthedocs.org/>`_ by using the git tag
that is the release version.


Update PyPI
-----------
The wheels and source should be uploaded to PyPI.

You should upload the wheels first, and the source formats last, to make sure
that pip users don't accidentally get a source install when they were
expecting a binary wheel. ::

    $ git clean -fxd  # to be safe
    $ python setup.py sdist --formats=gztar,zip  # to check
    # python setup.py sdist --formats=gztar,zip upload --sign

This will ask for your key PGP passphrase, in order to sign the built source
packages.


Upload files to github
----------------------

Once the wheels have been built without errors, go
to `<https://github.com/BYUCamachoLab/simphony/releases>`_, the main
Simphony repository in GitHub, and update the release by clicking ``Edit``
next to the appropriate release.

The subsequent page has are two locations to
add files and content, using an editable text window and as file uploads.

- Cut and paste the ``docs/changelog/0.3.0-changelog.md`` file contents into the text window.
- Upload ``dist/simphony-0.3.0.tar.gz`` as a binary file.
- Upload ``dist/simphony-0.3.0.zip`` as a binary file.
- Upload the file ``docs/changelog/0.3.0-changelog.md``.
- Hit the ``{Publish,Update} release`` button at the bottom.
