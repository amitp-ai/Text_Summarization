#setup.py
from setuptools import setup
from setuptools import find_packages
#find_packages will find all folders with __init__.py inside the directory containing setup.py. Otherwise have to list them e.g. packages=['src', 'app', 'test'] vs packages=find_packages
# To look at files inside bdist and sdist: rm -rf build .eggs *.egg-info dist && python3 setup.py sdist bdist_wheel && unzip -l dist/*.whl && tar --list -f dist/*.tar.gz

setup(
    name="MLOps", #it will be better to rename this and src directory to 'textSumm'
    version="0.0.1",
    description="text summarizer",
    author="Amit Patel",
    author_email='amitpatel.gt@gmail.com',
    packages=find_packages(where='.'),
    license="MIT",
    install_requires=['torch>=1.8.1', 'wandb>=0.10.31', 'rouge>=1.0.0'],
    url = 'https://github.com/amitp-ai/Text_Summarization_UCSD/tree/main/MLOps',
    package_data = {'MLOps.logs': ['*.log'], 'MLOps.Data': ['*.json', '*.csv', 'Training/Dicts/*'], 
                    'MLOps.SavedModels': ['*.pt']}
    # include_package_data=True, #this will add all directories containing __init__.py (and everything inside them)
    # tests_require=['pytest'],
    # setup_requires=['flake8', 'pytest-runner'],    
    # entry_points={'console_scripts': ['textSummarizerApi=app.app']}
)
