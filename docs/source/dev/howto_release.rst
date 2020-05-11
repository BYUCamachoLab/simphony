This file gives an overview of what is necessary to build binary releases for
Simphony.


Supported platforms and versions
================================
`NEP 29`_ outlines which Python versions are supported; For the first half of
2020, this will be Python >= 3.6. We test NumPy against all these versions
every time we merge code to master.  Binary installers may be available for a
subset of these versions (see below).

OS X
----
We don't actively test Simphony on MacOS machines. However, since the code
is relatively simple and based purely on Python and Python packages, we assume
that a MacOS user can install Simphony from pip with relative ease.


Windows
-------
We build 32- and 64-bit wheels on Windows. While installations may work on 
other versions, only Windows 10 is supported and tested on.


Linux
-----
We don't provide prebuilt versions of Simphony for Linux. Installation via 
pip or from source is straightforward and reliable.


BSD / Solaris
-------------
No binaries are provided, but successful builds on Solaris and BSD have been
reported.


Tool chain
==========
We build all our wheels on cloud infrastructure - so this list of compilers is
for information and debugging builds locally.  See the ``.travis.yml`` and
``appveyor.yml`` scripts in the `numpy wheels`_ repo for the definitive source
of the build recipes. Packages that are available using pip are noted.


Building source archives and wheels
-----------------------------------
You will need write permission for numpy-wheels in order to trigger wheel
builds.

- Python(s) from `python.org <https://python.org>`_ or linux distro.
- cython (pip)
- virtualenv (pip)
- Paver (pip)
- pandoc `pandoc.org <https://www.pandoc.org>`_ or linux distro.
- numpy-wheels `<https://github.com/MacPython/numpy-wheels>`_ (clone)


Building docs
-------------
Building the documents requires the items listed below and reproduced in a
doc_requirements.txt file, all installable from pip.

- Sphinx
- NumPy
- SciPy
- Matplotlib
- IPython


Uploading to PyPI
-----------------
- terryfy `<https://github.com/MacPython/terryfy>`_ (clone).
- beautifulsoup4 (pip)
- delocate (pip)
- auditwheel (pip)
- twine (pip)


Virtualenv
----------
Virtualenv is a very useful tool to keep several versions of packages around.
It is also used in the Paver script to build the docs.


What is released
================

Wheels
------
We currently support Python 3.6-3.8 on Windows, OSX, and Linux

* Windows: 32-bit and 64-bit wheels built using Appveyor;
* OSX: x64_86 OSX wheels built using travis-ci;
* Linux: 32-bit and 64-bit Manylinux1 wheels built using travis-ci.

See the `numpy wheels`_ building repository for more detail.

.. _numpy wheels : https://github.com/MacPython/numpy-wheels


Other
-----
- Release Notes
- Changelog


Source distribution
-------------------
We build source releases in both .zip and .tar.gz formats.


Release process
===============

Agree on a release schedule
---------------------------
A typical release schedule is one beta, two release candidates and a final
release.  It's best to discuss the timing on the mailing list first, in order
for people to get their commits in on time, get doc wiki edits merged, etc.
After a date is set, create a new maintenance/x.y.z branch, add new empty
release notes for the next version in the master branch and update the Trac
Milestones.


Make sure current branch builds a package correctly
---------------------------------------------------
::

    git clean -fxd
    python setup.py bdist
    python setup.py sdist

To actually build the binaries after everything is set up correctly, the
release.sh script can be used. For details of the build process itself, it is
best to read the pavement.py script.

.. note:: The following steps are repeated for the beta(s), release
   candidates(s) and the final release.


Check deprecations
------------------
Before the release branch is made, it should be checked that all deprecated
code that should be removed is actually removed, and all new deprecations say
in the docstring or deprecation warning at what version the code will be
removed.


Check the release notes
-----------------------
Use `towncrier`_ to build the release note and
commit the changes. This will remove all the fragments from
``doc/release/upcoming_changes`` and add ``doc/release/<version>-note.rst``.
Note that currently towncrier must be installed from its master branch as the
last release (19.2.0) is outdated.

    towncrier --version "<version>"
    git commit -m"Create release note"

Check that the release notes are up-to-date.

Update the release notes with a Highlights section. Mention some of the
following:

  - major new features
  - deprecated and removed features
  - supported Python versions
  - for SciPy, supported NumPy version(s)
  - outlook for the near future

.. _towncrier: https://github.com/hawkowl/towncrier


Update the release status and create a release "tag"
----------------------------------------------------
Identify the commit hash of the release, e.g. 1b2e1d63ff.

::
    git co 1b2e1d63ff # gives warning about detached head

First, change/check the following variables in ``pavement.py`` depending on the
release version::

    RELEASE_NOTES = 'doc/release/1.7.0-notes.rst'
    LOG_START = 'v1.6.0'
    LOG_END = 'maintenance/1.7.x'

Do any other changes. When you are ready to release, do the following
changes::

    diff --git a/setup.py b/setup.py
    index b1f53e3..8b36dbe 100755
    --- a/setup.py
    +++ b/setup.py
    @@ -57,7 +57,7 @@ PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-
     MAJOR               = 1
     MINOR               = 7
     MICRO               = 0
    -ISRELEASED          = False
    +ISRELEASED          = True
     VERSION             = '%d.%d.%drc1' % (MAJOR, MINOR, MICRO)

     # Return the git revision as a string

And make sure the ``VERSION`` variable is set properly.

Now you can make the release commit and tag.  We recommend you don't push
the commit or tag immediately, just in case you need to do more cleanup. We
prefer to defer the push of the tag until we're confident this is the exact
form of the released code (see: :ref:`push-tag-and-commit`):

    git commit -s -m "REL: Release." setup.py
    git tag -s <version>

The ``-s`` flag makes a PGP (usually GPG) signed tag.  Please do sign the
release tags.

The release tag should have the release number in the annotation (tag
message).  Unfortunately, the name of a tag can be changed without breaking the
signature, the contents of the message cannot.

See: https://github.com/scipy/scipy/issues/4919 for a discussion of signing
release tags, and https://keyring.debian.org/creating-key.html for instructions
on creating a GPG key if you do not have one.

To make your key more readily identifiable as you, consider sending your key
to public keyservers, with a command such as::

    gpg --send-keys <yourkeyid>


Update the version of the master branch
---------------------------------------
Increment the release number in setup.py. Release candidates should have "rc1"
(or "rc2", "rcN") appended to the X.Y.Z format.

Also create a new version hash in cversions.txt and a corresponding version
define NPY_x_y_API_VERSION in numpyconfig.h


Trigger the wheel builds on travis-ci and Appveyor
--------------------------------------------------
See the `numpy wheels` repository.

In that repository edit the files:

- ``.travis.yml``;
- ``appveyor.yml``.

In both cases, set the ``BUILD_COMMIT`` variable to the current release tag -
e.g. ``v1.11.1``.

Make sure that the release tag has been pushed.

Trigger a build by doing a commit of your edits to ``.travis.yml`` and
``appveyor.yml`` to the repository::

    cd /path/to/numpy-wheels
    # Edit .travis.yml, appveyor.yml
    git commit
    git push

The wheels, once built, appear at a Rackspace container pointed at by:

- http://wheels.scipy.org
- https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com

The HTTP address may update first, and you should wait 15 minutes after the
build finishes before fetching the binaries.


Make the release
----------------
Build the changelog and notes for upload with::

    paver write_release


Build and archive documentation
-------------------------------
Do::

    cd doc/
    make dist

to check that the documentation is in a buildable state. Then, after tagging,
create an archive of the documentation in the numpy/doc repo::

    # This checks out github.com/numpy/doc and adds (``git add``) the
    # documentation to the checked out repo.
    make merge-doc
    # Now edit the ``index.html`` file in the repo to reflect the new content,
    # and commit the changes
    git -C build/merge commit -am "Add documentation for <version>"
    # Push to numpy/doc repo
    git -C build/merge push


Update PyPI
-----------
The wheels and source should be uploaded to PyPI.

You should upload the wheels first, and the source formats last, to make sure
that pip users don't accidentally get a source install when they were
expecting a binary wheel.

You can do this automatically using the ``wheel-uploader`` script from
https://github.com/MacPython/terryfy.  Here is the recommended incantation for
downloading all the Windows, Manylinux, OSX wheels and uploading to PyPI. ::

    NPY_WHLS=~/wheelhouse   # local directory to cache wheel downloads
    CDN_URL=https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com
    wheel-uploader -u $CDN_URL -w $NPY_WHLS -v -s -t win numpy 1.11.1rc1
    wheel-uploader -u $CDN_URL -w warehouse -v -s -t macosx numpy 1.11.1rc1
    wheel-uploader -u $CDN_URL -w warehouse -v -s -t manylinux1 numpy 1.11.1rc1

The ``-v`` flag gives verbose feedback, ``-s`` causes the script to sign the
wheels with your GPG key before upload. Don't forget to upload the wheels
before the source tarball, so there is no period for which people switch from
an expected binary install to a source install from PyPI.

There are two ways to update the source release on PyPI, the first one is::

    $ git clean -fxd  # to be safe
    $ python setup.py sdist --formats=gztar,zip  # to check
    # python setup.py sdist --formats=gztar,zip upload --sign

This will ask for your key PGP passphrase, in order to sign the built source
packages.

The second way is to upload the PKG_INFO file inside the sdist dir in the
web interface of PyPI. The source tarball can also be uploaded through this
interface.

.. _push-tag-and-commit:


Push the release tag and commit
-------------------------------
Finally, now you are confident this tag correctly defines the source code that
you released you can push the tag and release commit up to github::

    git push  # Push release commit
    git push upstream <version>  # Push tag named <version>

where ``upstream`` points to the main https://github.com/numpy/numpy.git
repository.
