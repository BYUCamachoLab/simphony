.. _documenting:


A Guide to Simphony Documentation
=================================

When using `Sphinx <http://www.sphinx-doc.org/>`__ in combination with the
numpy conventions, you should use the ``sphinx.ext.napolean`` extension so that your
docstrings will be handled correctly. For example, Sphinx will extract the
``Parameters`` section from your docstring and convert it into a field
list.  Using ``sphinx.ext.napolean`` will also avoid the reStructuredText errors produced
by plain Sphinx when it encounters numpy docstring conventions like
section headers (e.g. ``-------------``) that sphinx does not expect to
find in docstrings.

Note that for documentation within Simphony, it is not necessary to do
``import simphony`` at the beginning of an example.  However, some
sub-modules, such as ``library``, are not imported by default, and you have to
include them explicitly::

  import simphony.libraries.ebeam

after which you may use it::

  simphony.libraries.ebeam.ebeam_wg_integral_1550(...)

.. rubric::
    **All inline documentation should adhere to the numpydoc** `formatting standard`_.

.. _`formatting standard`: https://numpydoc.readthedocs.io/en/latest/format.html


Sometimes there is a class attribute that you'd like documented but because
it isn't a function, it has no docstring. Luckily, sphinx allows us to 
autodocument attributes like so: ::

  #: Indicates some unknown error.
  API_ERROR = 1

Using multiple ``#:`` lines before any assignment statement, or a single ``#:`` comment 
to the right of the statement, work effectively the same as docstrings on 
objects picked up by autodoc. This includes handling inline rST, and 
auto-generating an rST header for the variable name; there's nothing extra 
you have to do to make that work (thanks, 
`abarnert <https://stackoverflow.com/a/20227174/11530613>`_ for the tip!).

Many of the library components are best documented with an accompanying picture
for cross-referencing port names and location.
You can add images to your inline documentation using rST syntax as long as 
the actual image resides in the docs/source directory. References 
can be expressed as absolute paths with respect to the source directory 
(see `Stack Overflow <https://stackoverflow.com/a/45739603/11530613>`_ for more details).

In keeping with the numpydoc standard, class initialization parameters
should be documented in the class docstring, not under ``__init__()``.

Names of classes, objects, constants, etc. should typically be linked to the
more detailed documentation of the referred object.

Examples presented are prefixed with the Python prompt ``>>>``. 

The docs are written in reST. There is a nice syntax guide with guidelines that
we follow in the documentation 
[here](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html). The 
majority of the documentation is generated from python docstrings written using
the NumPy documentation format.

Building the Simphony API and reference docs
--------------------------------------------
We currently use Sphinx_ for generating the API and reference
documentation for Simphony.  
.. You will need Sphinx >= 2.2.0.

If you only want to get the documentation, note that pre-built
versions can be found at

    https://simphonyphotonics.readthedocs.io/

in several different formats.

.. _Sphinx: http://www.sphinx-doc.org/


Instructions
------------

Building the documentation requires the Sphinx extension
`plot_directive`, which is shipped with Matplotlib_. This Sphinx extension can
be installed by installing Matplotlib. You will also need Python>=3.6.

Since large parts of the main documentation are obtained from Simphony via
``import simphony`` and examining the docstrings, you will need to first build
Simphony, and install it so that the correct version is imported.

Simphony has dependencies on other Python projects. Be sure to install its
requirements, listed in ``requirements.txt``.

Note that you can eg. install Simphony to a temporary location and set
the PYTHONPATH environment variable appropriately.
Alternatively, if using Python virtual environments (via e.g. ``conda``,
``virtualenv`` or the ``venv`` module), installing Simphony into a
new virtual environment is recommended.
All of the necessary dependencies for building the Simphony docs can be installed
with::

    pip install -r doc_requirements.txt

Now you are ready to generate the docs, so write::

    make html

in the ``doc/`` directory. If all goes well, this will generate a
``build/html`` subdirectory containing the built documentation. 

Note that building the documentation on Windows is currently not actively
supported, though it should be possible. (See Sphinx_ documentation
for more information.)

To build the PDF documentation, do instead::

   make latex
   make -C build/latex all-pdf

You will need to have Latex installed for this, inclusive of support for
Greek letters.  For example, on Ubuntu xenial ``texlive-lang-greek`` and
``cm-super`` are needed.  Also ``latexmk`` is needed on non-Windows systems.

Instead of the above, you can also do::

   make dist

which will rebuild Simphony, install it to a temporary location, and
build the documentation in all formats. This will most likely again
only work on Unix platforms.

The documentation for Simphony distributed at 
https://simphonyphotonics.readthedocs.io/ in html and
pdf format is also built with ``make dist``.  See `HOWTO RELEASE`_ for details
on how to update https://simphonyphotonics.readthedocs.io/.

.. _Matplotlib: https://matplotlib.org/
.. _HOWTO RELEASE: https://simphonyphotonics.readthedocs.io/

.. FIXME: Update the link for HOWTO RELEASE

Sphinx extensions
-----------------

Simphony's documentation uses several Sphinx extensions. While the
code docstrings are written using the `numpydoc`_ standard, we
actually use Sphinx's built-in `napolean`_ extension to parse
our files. Napolean has been included in the standard Sphinx since
version 1.3, so no special parsing extensions are required to generate this
documentation.

.. _numpydoc: https://python.org/pypi/numpydoc
.. _napolean: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

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

Our workflow, based strongly on the NumPy project, is in :ref:`development-workflow
<development-workflow>`.
