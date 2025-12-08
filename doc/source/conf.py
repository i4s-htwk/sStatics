# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sStatics'
copyright = '2024'
author = 'Paul Brassel'
release = '0.0.0'
html_logo = 'images/sStatics_Logo.png'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'nbsphinx',
]
exclude_patterns = []
html_static_path = ['.']
html_css_files = ['custom.css']

# Define the global role :python:``.
rst_prolog = """
.. role:: python(code)
    :language: python
"""

pygments_style = 'stata-light'
autodoc_typehints = 'none'
autodoc_default_options = {
    'inherited-members': None,
    'show-inheritance': None,
}
numpydoc_show_class_members = False
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/2.1', None),
    'python': ('https://docs.python.org/3', None),
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'

nbsphinx_codecell_lexer = 'python3'  # falls nötig
nbsphinx_prompt_width = '0'            # keine Prompt-Nummern anzeigen
nbsphinx_execute = 'always'
