.. _howto-document:


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

.. Some features described in this document require a recent version of
.. ``numpydoc``. For example, the **Yields** section was added in
.. ``numpydoc`` 0.6.

Note that for documentation within Simphony, it is not necessary to do
``import simphony`` at the beginning of an example.  However, some
sub-modules, such as ``library``, are not imported by default, and you have to
include them explicitly::

  import simphony.library.ebeam

after which you may use it::

  simphony.library.ebeam.ebeam_wg_integral_1550(...)

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