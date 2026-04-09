# Configuration file for the Sphinx documentation builder.

project = 'cfr (v2026)'
author = ''
copyright = '2026, Feng Zhu'

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

html_logo = '../cfr-logo.jpg'

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/fzhu2e/cfr',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_fullscreen_button': False,
    'article_header_end': 'version-switcher-header.html, article-header-buttons.html',
}

html_static_path = ['_static']
