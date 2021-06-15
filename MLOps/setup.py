#setup.py
from setuptools import setup, find_packages
#find_packages will find all folders with __init__.py inside the directory containing setup.py. Otherwie have to list them e.g. ['src', 'app', 'test']

setup(
    name="src", 
    version="0.0.1",
    description="text summarizer",
    author="Amit Patel",
    author_email='amitpatel.gt@gmail.com',
    packages=['src'],
    license="MIT",
    install_requires=['torch>=1.8.1', 'wandb>=0.10.31', 'rouge>=1.0.0'],
    url = 'https://github.com/amitp-ai/Text_Summarization_UCSD/tree/main/MLOps'
    )
