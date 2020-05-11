This file contains a walkthrough of the Simphony 0.3.0 release on Linux.
The commands can be copied into the command line, but be sure to
replace 0.3.0 by the correct version.


Release  Walkthrough
====================

Note that in the code snippets below, ``upstream`` refers to the root repository on
github and ``origin`` to a fork in your personal account. You may need to make adjustments
if you have not forked the repository but simply cloned it locally. You can
also edit ``.git/config`` and add ``upstream`` if it isn't already present.


.. Double check release versions
.. -----------------------------

.. Edit the ``.travis.yml`` and ``.appveyor.yml`` files to make sure they have the
.. correct version, and put in the commit hash for the ``REL`` commit created
.. above for ``BUILD_COMMIT``, see the _example from `v1.14.3`::

..     $ gvim .travis.yml .appveyor.yml
..     $ git commit -a
..     $ git push upstream HEAD


Update Release documentation
----------------------------

The file ``docs/changelog/0.3.0-changelog.md`` should be updated to reflect
the final list of changes. For now, this is a manual process. Below is a 
template of what sections to include: ::

    ## [MAJOR.MINOR.PATCH] - YEAR-MM-D

    This section provides an overview/summary of the release.

    ### Added
    - [#<val>](<link-to-issue>) Description of what was added.

    ### Changed
    - List of changes.

    ### Removed
    - List of items removed, if any.


Prepare the release commit
--------------------------

Checkout the branch for the release, make sure it is up to date, and clean the
repository::

    $ git checkout master
    $ git pull upstream master
    $ git clean -xdfq

Sanity check::

    $ pytest

Push this release directly onto the end of the master branch. This
requires write permission to the Simphony repository::

    $ git push upstream master


Build source releases and wheels
--------------------------------

We use ``setuptools`` and ``wheel`` to package Simphony. Make sure you have
the latest version installed: ::

    python3 -m pip install --user --upgrade setuptools wheel

In the same directory as ``setup.py``, run the following command: ::

    python3 setup.py sdist --formats=gztar,zip bdist_wheel

It will create the ``dist`` directory and place within it the ``*.zip`` and ``*.tar.gz``
source releases, as well as the built distribution ``*.whl``. Since Simphony
is not OS-specific (at least for now), the single wheel should be good for 
any platform.


Tag the release
---------------

.. note::
   You will need at least `Write access`_ on the main Simphony repository to
   create a GitHub release for Simphony.

.. _Write access: https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository

Once the wheels have been built without errors, go
to `<https://github.com/BYUCamachoLab/simphony/releases>`_, the main
Simphony repository in GitHub, and draft a new release. The name of the release
should be the version number: ::

    v0.3.0

Ensure that the files in ``dist`` have the correct, matching versions.


Upload files to github
----------------------

There are two locations to
add files and content, using an editable text window and as file uploads.

- Cut and paste the ``docs/changelog/0.3.0-changelog.md`` file contents into the text window.
- Upload ``dist/simphony-0.3.0.tar.gz`` as a binary file.
- Upload ``dist/simphony-0.3.0.zip`` as a binary file.
- Upload the file ``docs/changelog/0.3.0-changelog.md``.
- Hit the ``{Publish,Update} release`` button at the bottom.


Upload to PyPI
--------------

Upload to PyPI using ``twine``. A recent version of ``twine`` of is needed.

.. code-block:: sh

    $ cd ../simphony
    $ python3 -m twine upload dist/*.whl
    $ python3 -m twine upload dist/numpy-1.14.5.zip  # Upload last.

If one of the commands breaks in the middle, which is not uncommon, you may
need to selectively upload the remaining files because PyPI does not allow the
same file to be uploaded twice. The source file should be uploaded last to
avoid synchronization problems if pip users access the files while this is in
process. Note that PyPI only allows a single source distribution, here we have
chosen the zip archive.


Upload documents to simphonyphotonics.readthedocs.io
----------------------------------------------------

Documentation in the repository when the version is tagged (released) should 
already be up to date. If you have maintainer priveleges on the Simphony 
ReadTheDocs page, you should add the new release to the Active Versions 
section using the git tag name.
