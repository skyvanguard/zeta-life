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

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
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
suppress_warnings = ['autosummary']

# Don't fail on missing references
nitpicky = False

# -- Options for Napoleon (Google/NumPy docstrings) --------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

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
]

# -- Source suffix -----------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
