.. _documenting:

Documenting
===========
This is a guide on how to build the documentation web pages
(like the ones you are reading right now) from the source 
available on the Simphony repository. This is useful when 
writing new documentation, so that you can see how your
documentation pages will look.

The docs pages are written in reST. Read the `syntax guide on
reST` for more. However, much of the documentation is 
auto-generated from python docstrings found inline with
Simphony code, using the NumPy documentation format.

Since the docs require the Simphony code to build, you will
first need to set up a Simphony development environment, as
described in the "Setting Up the Environment" section of
:doc:`developing`.

Once you have set up your Simphony environment, and you have
your virtual environment activated, we will need to install
`Sphinx`_. We use Sphinx for generating docs pages: you need
to install the latest versions of both the ``Sphinx`` and
``sphinx-rtd-theme`` packages with ``pip``.

Once Sphinx is installed, you can use the following in the
``simphony/docs`` directory:

.. code:: sh

  make html

This will build the documentation pages at 
``docs/build/html``. Open up any of the HTML pages in that
folder to view the documentation on your local machine.

.. note::
  Building the documentation on Windows is not currently
  supported. (See `Sphinx`_ documentation for more
  information.)

There are also other targets you can build, such as PDF
pages instead of HTML pages. Use:

.. code:: sh

  make help

to see the other targets available.


.. _Sphinx: http://www.sphinx-doc.org/
.. _syntax guide on reST: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html
