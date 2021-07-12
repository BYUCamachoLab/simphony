.. _developing:

Setting up and using your development environment
=================================================

.. _recommended-development-setup:


Recommended development setup
-----------------------------

The best way to get started with a clean development environment is by first
creating a :ref:`virtual environment <virtual-environments>` to isolate the 
development dependencies from your system's python packages.

Once that is set up, simply clone the repository into a directory of your 
choosing, and after making sure your virtual environment is activated, follow
the instructions for :ref:`building-in-place`.


.. _testing-builds:


Testing builds
--------------

Before submitting any pull requests, however, you should 
ensure that a pip installation of your updated package installs and functions 
properly. To test this, try installing your package locally by removing all 
installed versions of Simphony (by running `pip3 uninstall simphony` 
repeatedly until no installations remain) and running the following commands 
(from Simphony's toplevel directory): ::

    $ python3 setup.py sdist bdist_wheel
    $ pip3 install dist/simphony-[VERSION].tar.gz

.. note::

    Remember that all tests of Simphony should pass before committing your changes.

Using ``pytest`` (through tox) is the recommended approach to running tests 
(see :ref:`running-tests`).


.. _building-in-place:

Building in-place
-----------------

For development, you can set up an in-place build so that changes made to
``.py`` files have effect without rebuild. First, run::

    $ make install

This allows you to import the in-place built Simphony *from any location the
Python environment used to install it is activated*. If this is your system
Python, you will be able to use the Simphony package from any directory, 
although installing a development version in your system Python is not recommended.

Now editing a Python source file in Simphony allows you to immediately
test and use your changes (in ``.py`` files), without even restarting the
interpreter.


.. _virtual-environments: 

Using virtual environments
--------------------------

One simple way to set up a development version of Simphony in parallel with a
regular install is to install the released version in
site-packages (perhaps using pip) and set
up the development version in a virtual environment.  The 
`venv`_ module should be included as part of a standard Python 3 installation. 
Create your virtualenv (named simphony-dev here) with::

    $ python3 -m venv simphony-dev

Now, whenever you want to switch to the virtual environment, you can use the
command ``source simphony-dev/bin/activate`` in the directory you created
the virtual environment in, and ``deactivate`` to exit from the
virtual environment back to your previous shell.

.. _venv: https://docs.python.org/3/library/venv.html


.. _running-tests:

Running tests
-------------

Any code that you have contributed should also have accompanying tests. The
style we adhere to in Simphony is to include a `/tests` directory wherever
you are developing your module and writing tests using the `pytest framework`_.

You already have all the testing dependencies installed if you followed the 
recommended procedure for setting up your development environment.

To run all tests, simply execute::

    $ make test

from the toplevel directory. This will use Tox_ to set up the testing environment
and run all the pytest commands.

.. _pytest framework: https://docs.pytest.org/en/latest/
.. _Tox: https://tox.readthedocs.io/

Running individual test files can be useful; it can be much faster than running the
whole test suite.
This can be done with::

    $ pytest path_to_testfile/test_file.py


Rebuilding & cleaning the workspace
-----------------------------------

There is no need to rebuilding Simphony after making changes to code if the
installation procedure discussed in :ref:`building-in-place` is followed.  
After/while making changes, however, you may want to clean
the workspace.  The standard way of doing this is (*note: deletes any
uncommitted files!*)::

    $ git clean -xdf

When you want to discard all changes and go back to the last commit in the
repo, use one of::

    $ git checkout .
    $ git reset --har

Development process - summary
-----------------------------
1. If you are a first-time contributor:

   * Go to `https://github.com/BYUCamachoLab/simphony
     <https://github.com/BYUCamachoLab/simphony>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone https://github.com/your-username/simphony.git

   * Change the directory::

      cd simphony

   * Add the upstream repository::

      git remote add upstream https://github.com/BYUCamachoLab/simphony.git

   * Now, `git remote -v` will show two remote repositories named:

     - ``upstream``, which refers to the ``simphony`` repository
     - ``origin``, which refers to your personal fork

2. Develop your contribution:

   * Pull the latest changes from upstream::

      git checkout master
      git pull upstream master

   * Create a branch for the feature you want to work on. Use a sensible,
     "human-readable" name such as "monte-carlo-simualations"::

      git checkout -b monte-carlo-simualations

   * Commit locally as you progress (``git add`` and ``git commit``)
     Use a clear and meaningful commit message,
     write tests that fail before your change and pass afterward, run all the
     :ref:`tests locally<development-environment>`. Be sure to document any
     changed behavior in docstrings, keeping to the NumPy docstring
     :ref:`standard<howto-document>`.

3. To submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin monte-carlo-simualations

   * Enter your GitHub username and password (repeat contributors or advanced
     users can remove this step by connecting to GitHub with SSH).

   * Go to GitHub. The new branch will show up with a green Pull Request
     button. Make sure the title and message are clear, concise, and 
     self-explanatory. Then click the button to submit it.

4. Review process:

   * Reviewers (the other developers and interested community members) will
     write inline and/or general comments on your Pull Request (PR) to help
     you improve its implementation, documentation and style.  Every single
     developer working on the project has their code reviewed, and we've come
     to see it as friendly conversation from which we all learn and the
     overall code quality benefits.  Therefore, please don't let the review
     discourage you from contributing: its only aim is to improve the quality
     of project, not to criticize (we are, after all, very grateful for the
     time you're donating!).

   * To update your PR, make your changes on your local repository, commit,
     **run tests, and only if they succeed** push to your fork. As soon as
     those changes are pushed up (to the same branch as before) the PR will
     update automatically. If you have no idea how to fix the test failures,
     you may push your changes anyway and ask for help in a PR comment.

   * Various continuous integration (CI) services are triggered after each PR
     update to build the code, run unit tests, measure code coverage and check
     coding style of your branch. The CI tests must pass before your PR can be
     merged. If CI fails, you can find out why by clicking on the "failed"
     icon (red cross) and inspecting the build and test log. To avoid overuse
     and waste of this resource,
     :ref:`test your work<recommended-development-setup>` locally before
     committing.

   * A PR must be **approved** by at least one core team member before merging.
     Approval means the core team member has carefully reviewed the changes,
     and the PR is ready for merging.

5. Document changes

   Beyond changes to a functions docstring and possible description in the
   general documentation, if your change introduces any user-facing
   modifications they may need to be mentioned in the release notes.
   To ensure your change gets added to the release notes, be sure to mention 
   what is changing in your pull request.

   If your change introduces a deprecation, also make sure to include this 
   fact in the pull request.

6. Cross referencing issues

   If the PR relates to any issues, you can add the text ``xref gh-xxxx`` where
   ``xxxx`` is the number of the issue to github comments. Likewise, if the PR
   solves an issue, replace the ``xref`` with ``closes``, ``fixes`` or any of
   the other flavors `github accepts <https://help.github.com/en/articles/
   closing-issues-using-keywords>`_.

For a more detailed discussion, read on and follow the links at the bottom of
this page.

Divergence between ``upstream/master`` and your feature branch
--------------------------------------------------------------

If GitHub indicates that the branch of your Pull Request can no longer
be merged automatically, you have to incorporate changes that have been made
since you started into your branch. Our recommended way to do this is to
rebase on master.

Guidelines
----------

* All code should have tests (see `test coverage`_ below for more details).
* All code should be `documented <https://numpydoc.readthedocs.io/
  en/latest/format.html#docstring-standard>`_.
  

Stylistic Guidelines
--------------------

* Set up your editor to follow `PEP 8 <https://www.python.org/dev/peps/
  pep-0008/>`_ (remove trailing white space, no tabs, etc.).  Check code with
  flake8, and be aware that a pre-commit hook will run the autoformatter
  Black_ over all of your committed code.

.. _Black: https://black.readthedocs.io/


Test coverage
-------------

Pull requests (PRs) that modify code should either have new tests, or modify existing
tests to fail before the PR and pass afterwards. You should :ref:`run the tests
<development-environment>` before pushing a PR.

Running Simphony's test suite locally requires some additional packages, such as
``pytest``. The additional testing dependencies are listed
in ``test_requirements.txt`` in the top-level directory, and can conveniently
be installed with::

    pip install -r test_requirements.txt

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure the test coverage, install
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`__
and then run::

  $ pytest --cov=simphony tests/


Development workflow
--------------------
**Prerequisites**:
You already have your own forked copy of the Simphony_ repository, you have 
configured git, and have `linked the upstream repository`_.

What is described below is a recommended workflow with Git.

.. _Simphony: https://github.com/BYUCamachoLab/simphony
.. _linked the upstream repository: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork

Basic workflow
##############

In short:

1. Start a new *feature branch* for each set of edits that you do.
   See :ref:`below <making-a-new-feature-branch>`.

2. Hack away! See :ref:`below <editing-workflow>`

3. When finished, push your feature branch to your own Github repo, and
   :ref:`create a pull request <asking-for-merging>`.

This way of working helps to keep work well organized and the history
as clear as possible.


For Developers
##############

This package is available on PyPI and updates are regularly pushed as "minor" 
or "micro" (patch) versions. Before submitting any pull requests, however, you should 
ensure that a pip installation of your updated package installs and functions 
properly. To test this, try installing your package locally by removing all 
installed versions of Simphony (by running ```pip3 uninstall simphony``` 
repeatedly until no installations remain) and running the following commands 
(from Simphony's toplevel directory): ::

   $ python3 setup.py sdist bdist_wheel  
   $ pip3 install dist/simphony-[VERSION].tar.gz

Also be sure to include tests for all the code you add, making certain the tests are
thorough and always pass.

For Maintainers
###############

Remember that all changes are to be integrated through pull requests. Development
work should be done in branches or forks of the repository. Once implemented 
(and tested on their own), these pull requests should be merged into the 
"master" branch for full testing with the whole program. 

- master (integration and final testing)
- feature-name (feature development and bug fixes)

Even if you are the lone developer, follow the methodology 
[here](https://softwareengineering.stackexchange.com/a/294048).

Be sure to update the version number manually before pushing each new version 
to PyPI. Also be sure to update the changelog.

Eventually, as the project grows, we will work up to using the methods 
detailed below, retained for future reference.


.. _making-a-new-feature-branch:

Making a new feature branch
---------------------------
First, fetch new commits from the ``upstream`` repository:

::

   git fetch upstream

Then, create a new branch based on the master branch of the upstream
repository::

   git checkout -b my-new-feature upstream/master


.. _editing-workflow:

The editing workflow
--------------------
Overview
--------

::

   # hack hack
   git status # Optional
   git diff # Optional
   git add modified_file
   git commit
   # push the branch to your own Github repo
   git push origin my-new-feature

In more detail
--------------

#. Make some changes. When you feel that you've made a complete, working set
   of related changes, move on to the next steps.

#. Optional: Check which files have changed with ``git status``.  
   You'll see a listing like this one::

     # On branch my-new-feature
     # Changed but not updated:
     #   (use "git add <file>..." to update what will be committed)
     #   (use "git checkout -- <file>..." to discard changes in working directory)
     #
     #	modified:   README
     #
     # Untracked files:
     #   (use "git add <file>..." to include in what will be committed)
     #
     #	INSTALL
     no changes added to commit (use "git add" and/or "git commit -a")

#. Optional: Compare the changes with the previous version using with ``git
   diff``. This brings up a simple text browser interface that
   highlights the difference between your files and the previous version.

#. Add any relevant modified or new files using  ``git add modified_file``. 
   This puts the files into a staging area, which is a queue
   of files that will be added to your next commit. Only add files that have
   related, complete changes. Leave files with unfinished changes for later
   commits.

#. To commit the staged files into the local copy of your repo, do ``git
   commit``. At this point, a text editor will open up to allow you to write a
   commit message. After saving
   your message and closing the editor, your commit will be saved. For trivial
   commits, a short commit message can be passed in through the command line
   using the ``-m`` flag.

#. Push the changes to your forked repo on GitHub::

      git push origin my-new-feature

.. note::

   Assuming you have followed the instructions in these pages, git will create
   a default link to your GitHub repo called ``origin``.  In git >= 1.7 you
   can ensure that the link to origin is permanently set by using the
   ``--set-upstream`` option::

      git push --set-upstream origin my-new-feature

   From now on git will know that ``my-new-feature`` is related to the
   ``my-new-feature`` branch in your own github repo. Subsequent push calls
   are then simplified to the following::

      git push

   You have to use ``--set-upstream`` for each new branch that you create.


It may be the case that while you were working on your edits, new commits have
been added to ``upstream`` that affect your work. In this case, follow the
:ref:`rebasing-on-master` section of this document to apply those changes to
your branch.


.. _asking-for-merging:

Asking for your changes to be merged with the main repo
-------------------------------------------------------
When you feel your work is finished, you can create a pull request (PR).

If your changes involve modifications to the API or addition/modification of a
function, you should be sure to emphasize this in the pull request. This may generate
changes and feedback. It might be prudent to start with this step if your
change may be controversial or make existing scripts not backward-compatible.

.. _rebasing-on-master:

Rebasing on master
------------------
This updates your feature branch with changes from the upstream `Simphony
github`_ repo. If you do not absolutely need to do this, try to avoid doing
it, except perhaps when you are finished. The first step will be to update
the remote repository with new commits from upstream::

   git fetch upstream

Next, you need to update the feature branch::

   # go to the feature branch
   git checkout my-new-feature
   # make a backup in case you mess up
   git branch tmp my-new-feature
   # rebase on upstream master branch
   git rebase upstream/master

If you have made changes to files that have changed also upstream,
this may generate merge conflicts that you need to resolve. See
:ref:`below<recovering-from-mess-up>` for help in this case.

Finally, remove the backup branch upon a successful rebase::

   git branch -D tmp

.. _Simphony github: https://github.com/BYUCamachoLab/simphony

.. note::

   Rebasing on master is preferred over merging upstream back to your
   branch. Using ``git merge`` and ``git pull`` is discouraged when
   working on feature branches.

.. _recovering-from-mess-up:

Recovering from mess-ups
------------------------
Sometimes, you mess up merges or rebases. Luckily, in Git it is
relatively straightforward to recover from such mistakes.

If you mess up during a rebase::

   git rebase --abort

If you notice you messed up after the rebase::

   # reset branch back to the saved point
   git reset --hard tmp

If you forgot to make a backup branch::

   # look at the reflog of the branch
   git reflog show my-feature-branch

   8630830 my-feature-branch@{0}: commit: BUG: io: close file handles immediately
   278dd2a my-feature-branch@{1}: rebase finished: refs/heads/my-feature-branch onto 11ee694744f2552d
   26aa21a my-feature-branch@{2}: commit: BUG: lib: make seek_gzip_factory not leak gzip obj
   ...

   # reset the branch to where it was before the botched rebase
   git reset --hard my-feature-branch@{2}

If you didn't actually mess up but there are merge conflicts, you need to
resolve those.  This can be one of the trickier things to get right.


Additional things you might want to do
######################################

Deleting a branch on GitHub
---------------------------
::

   git checkout master
   # delete branch locally
   git branch -D my-unwanted-branch
   # delete branch on github
   git push origin :my-unwanted-branch

(Note the colon ``:`` before ``test-branch``.  See also:
https://github.com/guides/remove-a-remote-branch

Releasing a Version
-------------------
.. _howto_release:

------------------------
How to Prepare a Release
------------------------

This file gives an overview of what is necessary to build binary releases for
Simphony.


Supported platforms and versions
--------------------------------
Platform Independent
--------------------

As a relatively straightforward Python-only package, Simphony doesn't require
any special builds for different operating systems and should be universally 
installable via pip.


Tool chain
----------
We build all our wheels using Linux.

The release process is also handled by GitHub Actions. Any time a release is
triggered (via the bash script in ``scripts/release``), the GitHub Action 
automatically builds the package, uploads it to PyPI, creates a GitHub release
(including source files), and the documentation automatically updates to the
most recent stable release.


What is released
----------------
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
---------------

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

We use bump2version_ to manage the version number throughout the project.
This provides 4 useful commands for managing the version number from the 
command line (``build``, ``patch``, ``minor``, ``major``, and ``release``) ::

    $ python -c "import simphony; print(simphony.__version__)"
    1.3.1

    $ bumpversion major; python -c "import simphony; print(simphony.__version__)"
    2.0.0-dev0

    $ bumpversion minor; python -c "import simphony; print(simphony.__version__)"
    2.1.0-dev0

    $ bumpversion patch; python -c "import simphony; print(simphony.__version__)"
    2.1.1-dev0
    
    $ bumpversion build; python -c "import simphony; print(simphony.__version__)"
    2.1.1-dev1

    $ bumpversion build; python -c "import simphony; print(simphony.__version__)"
    2.1.1-dev2

    $ bumpversion release; python -c "import simphony; print(simphony.__version__)"
    2.1.1

    $ bumpversion minor; python -c "import simphony; print(simphony.__version__)"
    2.2.0-dev0

.. _bump2version: https://github.com/c4urself/bump2version


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
--------------------
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


