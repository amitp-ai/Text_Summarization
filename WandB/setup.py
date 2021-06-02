#setup.py
from setuptools import setup, find_packages
#find packages will find all folders with __init__.py inside the directory containing setup.py
setup(
    name="src", 
    version="0.0.1",
    description="text summarizer",
    author="Amit Patel",
    packages=find_packages(),
    license="MIT"
    )
