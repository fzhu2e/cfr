# Configuration file for the Sphinx documentation builder.

project = 'cfr (CFRAME)'
author = ''
copyright = '2026, Feng Zhu, Julien Emile-Geay'

templates_path = ['_templates']

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx_design',
]

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/fzhu2e/cfr',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_fullscreen_button': False,
    'extra_footer': '<p><a href="../v2024/index.html">Switch to v2024 docs</a></p>',
}

html_static_path = ['_static']
