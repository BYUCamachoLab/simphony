.. _developing:

Maintaining and Developing
==========================
This will be a guide in how to contribute to Simphony by
writing code. We will go over how to set up a development
environment, testing, how to make a pull request, and how to
release a version of Simphony. This guide assumes you have a
good understanding of Python development and Git.


Setting Up the Environment
--------------------------
Simphony is a Python 3 package, and so we recommend using
the built-in ``venv`` module for Python 3 to
`create a virtual environment`_. From this point forwards,
all commands that we provide assume that your new virtual
environment is activated.

After creating and activating your virtual environment, 
go to the `Simphony Github page` and fork the project. You
should now have your own copy of the repository on Github.
Run:

.. code:: sh

  git clone https://github.com/[your username]/simphony.git


to download the Github repository onto your machine. You
will also want to run:

.. code:: sh

  git remote add upstream https://github.com/BYUCamachoLab/simphony.git

This will allow you to pull any changes made to the original
repository.

Once those steps are complete, run the command:

.. code:: sh

  make install

in the root directory of Simphony. This will build the 
development version of Simphony as a package in your virtual
environment. The build will update whenever you update
Simphony code on your machine, allowing you to use/import 
the development version in Python files.

You can now start writing your changes to Simphony code.
Make sure to always pull changes from upstream before
starting development.


Testing
-------
We expect code to be thoroughly tested before it gets merged
with the main Simphony repository.

In Simphony, we include a ``tests`` directory inside main
sections of code, where we write tests using the 
`pytest framework`_. You should write new tests in these 
test directories for any new code you develop.

Your code should pass all existing tests before submitting a
pull request. Any time you make a commit, a pre-commit hook
will run some basic tests and formatters. To run all tests,
run the command:

.. code:: sh

  make test

in the root directory of Simphony. You may also wish to run
an individual test, by using:

.. code:: sh

  pytest path_to_test/test_file.py

Specific formatting issues must also be checked before
changes will be merged. You can check all formatting issues
using:

.. code:: sh

  flake8

An additional step you can take with your tests is to
measure the test coverage of your code. To do this, install
`pytest-cov`_, then run the command:

.. code:: sh

  pytest --cov=simphony tests/

A final check you may perform before submitting any pull
requests is to ensure that your Simphony package will 
install properly. This is mainly necessary if you intend to
release a new version of Simphony. To test this, remove all 
current installations of Simphony (repeat the command
``pip uninstall simphony`` until no versions remain) and
then run the following commands from Simphony's root 
directory:

.. code:: sh

  python3 setup.py sdist bdist_wheel
  pip3 install dist/simphony-[VERSION].tar.gz

.. note::
  Doing this will require you to rebuild your development
  version of Simphony using ``make install`` again. Before
  you do this, remove the test version of Simphony you just
  installed using ``pip uninstall simphony`` once more.


Make a Pull Request
-------------------
After you've made your changes and have done the testing
steps above, you're ready to make a pull request. Push your
changes to your Github fork, and navigate to the fork's page
on Github. You should see a button that will create a pull
request for you.

Reviewers will look over your pull request before merging
your changes into the main repository, so we expect you to
write a clear and concise explanation of your changes
attached to your pull request. If your changes are
extensive, you will likely need to write more explanation,
and perhaps explain the motivation for such extensive
changes. Your description of the changes you've made will be
used when writing the release notes.

Reviewers will comment on any last minute changes they want
to see before merging, such as style and inline
documentation, so make sure your code is polished before
submitting a pull request. All documentation should follow
the `numpy doc formatting standard`_.

.. note::
  If you need to make changes while a pull request is still
  being reviewed, just push your changes to your fork.
  The pull request will automatically update to match.

When you submit a pull request, automatic tests will trigger
and run through Github's services. These must all pass
before your pull request will be accepted. If any fail,
click the red cross to pull up a test log, which will help
you find out why they failed. Ideally, you will have tested
your project before creating the pull request in the first
place, so that these tests will not fail.

Finally, when the reviewers believe that the pull request is
ready, they will approve the pull request and it will be
merged into the main repository.


Releasing
---------
Most contributors won't have to worry about the release
process, since this is up to the core development team. It
may still be informative, so we include it here.

The release process is handled by GitHub Actions. When the
release script is triggered, it builds the package for 
Python 3.6-3.8 for Windows, Mac and Linux. It uploads the
package to PyPI, creates a GitHub release, and updates the
documentation to the most recent stable release.

Before the release, all deprecated code should be removed,
and a changelog should be written for the new version. For
the changelog, follow the style of previous changelogs when
writing. The documentation should be ready for build, see
:doc:`documenting` for how to build. Use `bump2version`_
to update the version number throughout the project.

Once all of this is complete, run the bash script at
``scripts/release``, and the rest will be taken care of
automatically.


.. _create a virtual environment: https://docs.python.org/3/tutorial/venv.html
.. _pytest framework: https://docs.pytest.org/en/latest/
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _Simphony Github page: https://github.com/BYUCamachoLab/simphony
.. _`numpy doc formatting standard`: https://numpydoc.readthedocs.io/en/latest/format.html
.. _bump2version: https://github.com/c4urself/bump2version
