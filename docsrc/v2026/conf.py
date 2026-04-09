# Configuration file for the Sphinx documentation builder.

project = 'cfr (CFRAME)'
author = ''
copyright = '2026, Feng Zhu, Julien Emile-Geay'

# This version label is used by the version switcher dropdown
version = 'v2026'

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
    'switcher': {
        'json_url': 'https://fzhu2e.github.io/cfr/versions.json',
        'version_match': version,
    },
    'navbar_end': ['version-switcher', 'theme-switcher'],
}

html_static_path = ['_static']
