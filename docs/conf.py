# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'napari-dab-cellcount'
author = 'Jyotirmay,  ADAPTER HEAVILY from Cellpose-napari by Carsen Stringer & Marius Pachitariu'

# The full version, including alpha/beta/rc tags
release = '0.1.4'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon']
#extensions = ['sphinx.ext.autodoc',
#            'sphinx.ext.mathjax',
#            'sphinx.ext.viewcode',
#            'sphinx.ext.autosummary',
#            'sphinx.ext.doctest',
#            'sphinx.ext.inheritance_diagram',
#            'autoapi.extension',
#            'sphinx.ext.napoleon']

autoapi_dirs = ['../napari-dab-cellcount']

source_suffix='.rst'

master_doc = 'index'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
