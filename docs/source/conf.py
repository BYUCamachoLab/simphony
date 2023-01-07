# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
import re

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# -- Project information -----------------------------------------------------

project = "Simphony"
copyright = "2019-2021, Simphony Project Contributors"
author = "Sequoia Ploeg, et al."

# The full version, including alpha/beta/rc tags
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
release = '0.6.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # "sphinxext.rediraffe",
    "sphinx_design",
    "sphinx_copybutton",
    # "ablog",
    # "jupyter_sphinx",
    "matplotlib.sphinxext.mathmpl",
    "matplotlib.sphinxext.plot_directive",
    # "myst_nb",
    # "nbsphinx",  # Uncomment and comment-out MyST-NB for local testing purposes.
    "numpydoc",
    "sphinx_togglebutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    # 'IPython.sphinxext.ipython_console_highlighting',
    # 'IPython.sphinxext.ipython_directive',
    "sphinx.ext.imgmath",
    # 'sphinx.ext.mathjax', #
    # 'sphinx.ext.githubpages', #
    # 'sphinx_autodoc_typehints', #
]

# -- Internationalization ------------------------------------------------
# specifying the natural language populates some key tags
language = "en"


autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_typehints = "none"
autoclass_content = "class"  # Add __init__ doc (ie. params) to class summaries
# html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
# add_module_names = False # Remove namespaces from class/method signatures

autodoc_mock_imports = [
    "parsimonious",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints"]

# -- Extension options -------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
# myst_enable_extensions = ["colon_fence", "linkify", "substitution"]

panels_add_bootstrap_css = False

# -- Plot directive configuration --------------------------------------------

# For speedup, decide which plot_formats to build based on build targets:
#     html only -> png
#     latex only -> pdf
#     all other cases, including html + latex -> png, pdf
# For simplicity, we assume that the build targets appear in the command line.
# We're falling back on using all formats in case that assumption fails.
formats = {'html': ('png', 100), 'latex': ('pdf', 100)}
plot_formats = [formats[target] for target in ['html', 'latex']
                if target in sys.argv] or list(formats.values())

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# Configuration guide: https://pydata-sphinx-theme.readthedocs.io/en/v0.8.1/user_guide/configuring.html

html_theme = 'pydata_sphinx_theme'
# html_logo = "_static/gdsframeworklogo2.png"
# html_favicon = "_static/logo.svg"
# html_sourcelink_suffix = ""

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    # "header_links_before_dropdown": 4,
    # "icon_links": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/BYUCamachoLab/simphony",
    #         "icon": "fab fa-github-square",
    #         "type": "fontawesome",
    #     },
    # ],
    "github_url": "https://github.com/BYUCamachoLab/simphony",
    # "logo": {
    #     "text": "PyData Theme",
    #     "image_dark": "logo-dark.svg",
    #     "alt_text": "PyData Theme",
    # },
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    # "navbar_center": ["version-switcher", "navbar-nav"],
    # "announcement": "https://raw.githubusercontent.com/pydata/pydata-sphinx-theme/main/docs/_templates/custom-template.html",
    "show_nav_level": 1,
    # "navbar_start": ["navbar-logo"],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "footer_items": ["copyright", "sphinx-version"],
    # "secondary_sidebar_items": ["page-toc.html"],  # Remove the source buttons
    # "switcher": {
    #     "json_url": json_url,
    #     "version_match": version_match,
    # },
    # "search_bar_position": "navbar",  # TODO: Deprecated - remove in future version
    "external_links": [
      {"name": "Changelog", "url": "https://github.com/BYUCamachoLab/simphony/tree/master/docs/changelog"},
    ],
}

# Custom sidebar templates, maps document names to template names.
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "index": [],
    # "community/index": [
    #     "sidebar-nav-bs",
    #     "custom-template",
    # ],  # This ensures we test for custom sidebars
    # "examples/no-sidebar": [],  # Test what page looks like with no sidebar items
    # "examples/persistent-search-field": ["search-field"],
    # # Blog sidebars
    # # ref: https://ablog.readthedocs.io/manual/ablog-configuration-options/#blog-sidebars
    # "examples/blog/*": [
    #     "postcard.html",
    #     "recentposts.html",
    #     "tagcloud.html",
    #     "categories.html",
    #     "authors.html",
    #     "languages.html",
    #     "locations.html",
    #     "archives.html",
    # ],
}

html_context = {
    "github_url": "https://github.com", # or your GitHub Enterprise interprise
    "github_user": "BYUCamachoLab",
    "github_repo": "simphony",
    "github_version": "master",
    "doc_path": "docs/source",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# rediraffe_redirects = {
#     # "contributing.rst": "community/index.rst",
# }

# ABlog configuration
# blog_path = "examples/blog/index"
# blog_authors = {
#     "pydata": ("PyData", "https://pydata.org"),
#     "jupyter": ("Jupyter", "https://jupyter.org"),
# }

imgmath_image_format = "svg"

# The suffix of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    # '.txt': 'markdown',
    # '.md': 'markdown',
}

# The master toctree document.
master_doc = "index"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%B %d, %Y"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
# language = None

# List of documents that shouldn't be included in the build.
# unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, `todo` and `todoList` produce output, else they produce nothing.
# todo_include_todos = True

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True
# html_domain_indices = False # NumPy setting

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None
# html_file_suffix = '.html' # NumPy setting

# Output file base name for HTML help builder.

# html_use_modindex = True
# html_copy_source = False
htmlhelp_basename = "simphony"

if "sphinx.ext.pngmath" in extensions:
    pngmath_use_preview = True
    pngmath_dvipng_args = ["-gamma", "1.5", "-D", "96", "-bg", "Transparent"]

# plot_html_show_formats = False
# plot_html_show_source_link = False


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "Simphony.tex", "Simphony Documentation", "Sequoia Ploeg", "manual"),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "Simphony", "Simphony Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Simphony",
        "Simphony Documentation",
        author,
        "Simphony",
        "A simulator for photonic circuits.",
        "Miscellaneous",
    ),
]
