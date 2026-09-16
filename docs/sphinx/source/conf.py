import os
import sys

sys.path.insert(0, os.path.abspath('../../../'))

project = 'vLLM SUPA'
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
    'breathe',
]

breathe_projects = {'VLLM_SUPA': '../../doxygen/xml'}
breathe_default_project = 'VLLM_SUPA'

html_theme = 'sphinx_rtd_theme'
html_static_path = []
latex_engine = 'xelatex'
latex_documents = [
    ('index', 'vllm_supa_technical.tex', 'vLLM SUPA 技术文档', '壁仞科技', 'manual'),
]
