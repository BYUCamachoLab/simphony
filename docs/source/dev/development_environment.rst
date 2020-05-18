.. _development-environment:

Setting up and using your development environment
=================================================

.. _recommended-development-setup:


Recommended development setup
-----------------------------

For this chapter we assume that you have already set up your git repo.


.. _testing-builds:


Testing code
------------

Any code that you have contributed should also have accompanying tests. The
style we adhere to in Simphony is to include a `/tests` directory wherever
you are developing your module and writing tests using the `pytest framework`_.

To run all tests, simply execute

```
pytest
```

from the toplevel directory.

.. _pytest framework: https://docs.pytest.org/en/latest/


Testing builds
--------------

Before submitting any pull requests, however, you should 
ensure that a pip installation of your updated package installs and functions 
properly. To test this, try installing your package locally by removing all 
installed versions of Simphony (by running `pip3 uninstall simphony` 
repeatedly until no installations remain) and running the following commands 
(from Simphony's toplevel directory):

```
python3 setup.py sdist bdist_wheel
pip3 install dist/simphony-[VERSION].tar.gz
```

.. note::

    Remember that all tests of Simphony should pass before committing your changes.

Using ``pytest`` is the recommended approach to running tests (see :ref:`running-tests`).


.. _building-in-place:

Building in-place
-----------------

For development, you can set up an in-place build so that changes made to
``.py`` files have effect without rebuild. First, run::

    $ pip install -e .

This allows you to import the in-place built Simphony *from any location the
Python environment used to install it is activated*. If this is your system
Python, you will be able to use the Simphony package from any directory.

Now editing a Python source file in Simphony allows you to immediately
test and use your changes (in ``.py`` files), without even restarting the
interpreter.


Using virtual environments
--------------------------

A frequently asked question is "How do I set up a development version of NumPy
in parallel to a released version that I use to do my job/research?".

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

To install testing dependencies, run::

    $ pip install .[test]

To run all tests, simply execute::

    $ pytest

from the toplevel directory. Note that pytest must be run under a Python 3
environment.

.. _pytest framework: https://docs.pytest.org/en/latest/

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
    $ git reset --hard
