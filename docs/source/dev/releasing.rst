===================
Releasing a Version
===================

.. _howto_release:

------------------------
How to Prepare a Release
------------------------

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

We build all our wheels using Linux.

The release process is also handled by GitHub Actions. Any time a release is
triggered (via the bash script in ``scripts/release``), the GitHub Action 
automatically builds the package, uploads it to PyPI, creates a GitHub release
(including source files), and the documentation automatically updates to the
most recent stable release.


What is released
================

Wheels
------
We currently support Python 3.6-3.8 on Windows, OSX, and Linux.

Simphony ought to be OS-independent. Hence, a ``none-any`` built wheel is 
included in the release.


Other
-----
- Changelog


Source distribution
-------------------
We build source releases in ``.tar.gz`` format. 


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
to ``master``. Also make sure that you have the master branch checked out, as
all releases are created from ``master``. ::

    git checkout master


Update the version of the master branch
---------------------------------------
Increment the version number anywhere that needs it. 
Release candidates should have "rc1"
(or "rc2", "rcN") appended to the X.Y.Z format.


Build and archive documentation
-------------------------------
Do::

    cd docs/
    make html

to check that the documentation is in a buildable state. Later, when you 
trigger the release action, the documentation on ReadTheDocs will 
automatically be built and updated to the most recent stable version.


Trigger the Release Action
--------------------------

Run the bash script included in ``scripts/release``. This will automatically
create a tag at the current revision history and push it to GitHub. You may
need to provide your GitHub credentials. 

The script will ensure that a changelog for the current version exists. If it
does not, the script will fail and terminate without tagging.


-----------------------
Step-by-Step Directions
-----------------------

This file contains a walkthrough of the Simphony 0.3.0 release on Linux.
The commands can be copied into the command line, but be sure to
replace 0.3.0 by the correct version.


Release  Walkthrough
====================

Note that in the code snippets below, ``upstream`` refers to the root repository on
github and ``origin`` to a fork in your personal account. You may need to make adjustments
if you have not forked the repository but simply cloned it locally. You can
also edit ``.git/config`` and add ``upstream`` if it isn't already present.


Update Release documentation
----------------------------

The file ``docs/changelog/0.3.0-changelog.md`` should be updated to reflect
the final list of changes. For now, this is a manual process. Below is the actual
v0.3.0 changelog, showing what sections to include: 

.. code-block:: md

    ## [0.3.0] - 2020-05-18

    This version is a complete codebase rewrite implementing a much more
    human-friendly way of defining circuits and running simulations. 

    Circuits are now defined and stored in a clear, easy-to-understand 
    object-oriented way. Integration with other packages should be easy, and the
    creation of model libraries or other things that extend the functionality of
    Simphony should fit easily into the existing framework.

    NOTE: THIS VERSION IS NOT BACKWARDS COMPATIBLE.

    ### Added
    - [#31](https://github.com/sequoiap/simphony/issues/31) Simphony was placed 
        under continuous integration to ensure code health.
    - [#11](https://github.com/BYUCamachoLab/simphony/issues/11) Examples and 
        documentation were created and are hosted online at 
        https://simphonyphotonics.readthedocs.io/
    - Nearly the entire SiEPIC EBeam PDK Compact Model Library is available.
    - Parser compatible with circuit files exported by SiEPIC-Tools in KLayout
        allows Simphony to perform simulations on files created in KLayout.

    ### Changed
    - Entire package was rewritten from previous versions.

    ### Removed
    - Simphony simulation engine completely rewritten. Former scripts are not 
        compatible with this version and will need to be rewritten.



Prepare the release commit
--------------------------

Checkout the master branch for release and make sure it is up to date. ::

    $ git checkout master
    $ git pull upstream master

Sanity check::

    $ pytest

If there are any uncommitted changes, push them directly onto the end of the 
master branch. This requires write permission to the Simphony repository::

    $ git push upstream master


Build source releases and wheels
--------------------------------

.. note:: 
   Simphony gets published automatically to PyPI when a new version is tagged
   in GitHub. The following is the process followed by the GitHub Actions workflow
   to publish to PyPI, and does NOT need to be performed manually.

We use ``setuptools`` and ``wheel`` to package Simphony. Make sure you have
the latest version installed: ::

    python3 -m pip install --user --upgrade setuptools wheel

In the same directory as ``setup.py``, run the following command: ::

    python3 setup.py sdist --formats=gztar,zip bdist_wheel

It will create the ``dist`` directory and place within it the ``*.zip`` and ``*.tar.gz``
source releases, as well as the built distribution ``*.whl``. Since Simphony
is not OS-specific (at least for now), the single wheel should be good for 
any platform.


Trigger the release
-------------------

.. note::
   You will need at least `Write access`_ on the main Simphony repository to
   create a GitHub release for Simphony.

.. _Write access: https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository

.. warning::
   Make absolutely sure the version numbers are all up to date throughout
   the package.

Note that all these commands are executed in Bash terminal on Linux.

Simply run the script ``scripts/release`` to trigger a release. It will collect
all the required files and parse the version number automatically, creating
a git tag and pushing to remote. You may need to provide your GitHub credentials
in the terminal. ::

    ./scripts/release


Check documentation at simphonyphotonics.readthedocs.io
-------------------------------------------------------

Documentation in the repository when the version is tagged (released) should 
already be up to date. If you have maintainer priveleges on the Simphony 
ReadTheDocs page, you should add the new release to the Active Versions 
section using the git tag name.


