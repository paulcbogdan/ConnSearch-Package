import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info[:3] < (3, 5, 0):
    print("Requires Python 3.5 to run.")
    sys.exit(1)

desc = '`connsearch` is a Python package for analysis of functional connectivity data.\n' \
       'It is premised on dividing the connectome into network components and ' \
       'fitting an independent model for each component.\n' \
       'See our paper: Bogdan, P.C., Iordan, A. D., Shobrook, J., & Dolcos, F. (2023). ' \
       'ConnSearch: A Framework for Functional Connectivity Analysis Designed ' \
       'for Interpretability and Effectiveness at Limited Sample Sizes. ' \
       'NeuroImage, 120274.'

fp_install_requires = 'requirements.txt'
with open(fp_install_requires) as f:
    install_requires = f.read().splitlines()
setup(
    name="connsearch",
    description="Analysis of connectivity data by dividing the connectome and fitting independent models",
    long_description=desc,
    long_description_content_type="text/markdown",
    version="v0.0.12",
    packages=["connsearch", "connsearch.permute", "connsearch.report"],
    python_requires=">=3.5",
    url="https://github.com/paulcbogdan/ConnSearch_pkg",
    author="paulcbogdan",
    author_email="paulcbogdan@gmail.com",
    install_requires=install_requires,
    keywords=["fmri", "connectivity", "multivariate", "machine learning"],
    license="MIT"
)