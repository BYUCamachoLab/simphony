########################
Contributing to Simphony
########################

Not a coder? Not a problem! Simphony is still a developing project, 
and we could use a lot of help.
These are all activities we'd like to get help with (they're all important, so
we list them in alphabetical order):

- Code maintenance and development (architecture input welcome)
- Fundraising
- Marketing
- Writing technical documentation and examples

Development process - summary
=============================

Here's the short summary, complete TOC links are below:

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
     button. Make sure the title and message are clear, concise, and self-
     explanatory. Then click the button to submit it.

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
  pyflakes / flake8.
  .. FIXME: Do we want to use `black` instead?


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

Building docs
-------------

To build docs, run ``make`` from the ``docs`` directory. ``make help`` lists
all targets. For example, to build the HTML documentation, you can run:

.. code:: sh

    make html

Then, all the HTML files will be generated in ``docs/build/html/``.
Since the documentation is based on docstrings, the appropriate version of
Simphony must be installed in the host python used to run sphinx.

Requirements
~~~~~~~~~~~~

`Sphinx <http://www.sphinx-doc.org/en/stable/>`__ is needed to build
the documentation.

These additional dependencies for building the documentation are listed in
``doc_requirements.txt`` and can be conveniently installed with::

    pip install -r doc_requirements.txt

The documentation includes mathematical formulae with LaTeX formatting.
A working LaTeX document production system 
(e.g. `texlive <https://www.tug.org/texlive/>`__) is required for the
proper rendering of the LaTeX math in the documentation.


Development process - details
=============================

.. toctree::
   :maxdepth: 2

   development_environment
   development_workflow
   docs/howto_document
   docs/howto_build_docs
   releasing
   bugs

Our workflow, based strongly on the NumPy project, is in :ref:`development-workflow
<development-workflow>`.
