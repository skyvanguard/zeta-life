# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'Zeta-Life'
copyright = '2026, IPUESA Research'
author = 'IPUESA Research'
release = '0.1.0'
version = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

autosummary_generate = True

# Mock imports that may cause issues during doc generation
autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.optim',
    'torch.nn.functional',
    'mpmath',
    'scipy',
    'scipy.stats',
    'scipy.signal',
    'PIL',
    'cv2',
]

# Suppress specific warnings
suppress_warnings = ['autosummary', 'myst.xref_missing']

# Don't fail on missing references
nitpicky = False

# -- Options for Napoleon (Google/NumPy docstrings) --------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = "Zeta-Life"

# PyData theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ipuesa/zeta-life",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "logo": {
        "text": "Zeta-Life",
    },
    "announcement": "Research framework connecting Riemann zeta mathematics with emergent intelligence",
}

html_context = {
    "github_user": "ipuesa",
    "github_repo": "zeta-life",
    "github_version": "master",
    "doc_path": "docs",
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- MyST Parser options -----------------------------------------------------
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'amsmath',
]

myst_heading_anchors = 3

# -- Source suffix -----------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
