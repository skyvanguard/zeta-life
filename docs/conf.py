# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'Zeta-Life'
copyright = '2026, Francisco Ruiz'
author = 'Francisco Ruiz'
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
exclude_patterns = ['_build', '_build_*', 'Thumbs.db', '.DS_Store']

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
    'torch', 'torch.nn', 'torch.optim', 'torch.nn.functional',
    'mpmath', 'scipy', 'scipy.stats', 'scipy.signal', 'PIL', 'cv2',
]

# Suppress specific warnings
suppress_warnings = ['autosummary', 'myst.xref_missing']
nitpicky = False

# -- Options for Napoleon (Google/NumPy docstrings) --------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = "Zeta-Life"
html_favicon = "_static/logo.svg"

# PyData theme options - Professional navbar configuration (inspired by NumPy/sklearn)
html_theme_options = {
    # Header/Navbar - Clean and minimal
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navbar_align": "content",

    # Navigation
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "collapse_navigation": True,

    # Sidebar - Clean
    "primary_sidebar_end": [],
    "secondary_sidebar_items": ["page-toc"],

    # Footer
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": ["sphinx-version"],

    # Appearance
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",

    # Logo
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
        "text": "Zeta-Life",
        "alt_text": "Zeta-Life - Home",
    },

    # Version switcher
    "switcher": {
        "json_url": "https://fruizvillar.github.io/zeta-life/_static/switcher.json",
        "version_match": version,
    },

    # Icon links (top right) - Only GitHub
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fruizvillar/zeta-life",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],

    # Header links - Maximum 5 visible items before dropdown
    "header_links_before_dropdown": 5,

    # Clean UI
    "use_edit_page_button": False,
    "show_version_warning_banner": False,
    "article_header_start": ["breadcrumbs"],
    "article_header_end": [],
}

# Context for templates
html_context = {
    "github_user": "fruizvillar",
    "github_repo": "zeta-life",
    "github_version": "master",
    "doc_path": "docs",
    "default_mode": "auto",
}

# Custom CSS files
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
    'substitution',
]

myst_heading_anchors = 3

# -- Source suffix -----------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
