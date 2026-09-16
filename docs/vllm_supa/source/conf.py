import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath('../../../'))

project = 'vLLM SUPA 用户指南'
copyright = '2026, 壁仞科技'
author = '壁仞科技'
release = '0.27.1'
master_doc = 'index'
language = 'zh_CN'
templates_path = ['_templates']
exclude_patterns = []

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
]

latex_use_xindy = False
from br_latex_conf import (  # noqa: F401,E402
    latex_additional_files,
    latex_elements,
    latex_engine,
    latex_secnum_depth,
    latex_toc_depth,
)

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
latex_documents = [
    ('index', 'vllm_supa_user_guide.tex', '壁仞™ vLLM SUPA 用户指南', '壁仞科技', 'manual'),
]
