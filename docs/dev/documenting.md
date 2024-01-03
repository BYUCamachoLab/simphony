(documenting)=
# Documenting

This is a guide on how to build the documentation web pages (like the ones you are reading right now) from the source
available on the Simphony repository. This is useful when writing new documentation, so that you can see how your
documentation pages will look.

The docs pages are written mostly in Markdown. Read the [syntax guide on
MyST](https://myst-parser.readthedocs.io/en/latest/) for more. The API documentation is auto-generated from python
docstrings found inline with Simphony code, using the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
format.

Since the docs require the Simphony code to build, you will first need to set up a Simphony development environment, as
described in the "Setting Up the Environment" section of [](developing.md).

We use Jupyter Book and Sphinx for generating documentation pages. By following the previously mentioned steps to set up
your Simphony environment, these packages are already installed.

Once Sphinx is installed, you can use the Makefile to build the docs:

```bash
make doc
```

This will build the documentation pages at ``docs/build/html``. Open up any of the HTML pages in that folder to view the
documentation on your local machine, or use the Makefile target:

```bash
make serve
```

<!-- ```{note}
Building the documentation on Windows is not currently supported. (See `Sphinx`_ documentation for more information.)
``` -->
