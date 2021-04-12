from setuptools import setup

def read():
    with open('README.md') as fh:
        return fh.read()

setup(
    name='ModelBuilding',
    version='0.11dev',
    description='Building Models for Text Summarization',
    # long_description=read(),
    packages=['ModelBuilding'], #, 'DataCollection', 'DataWrangling', 'LiteratureSurvey'],
)


'''
setup(
    name='Text_Summarizer',
    version='0.1dev',
    description='Building Models for Text Summarization',
    long_description=read(),
    packages=['ModelBuilding'], #, 'DataCollection', 'DataWrangling', 'LiteratureSurvey'],
    install_requires=[
        'pytest>=4.3.1',
        'pytest-runner>=4.4',
        'click>=7.0'], #for command line
    scripts = [], #list of .py files that are needed but aren't specified in packages=[]
    package_data=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    entry_points={'console_scripts': ['text_summarization_model=Model_Building.command_line:main'],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/amitp-ai/Text_Summarization_UCSD',
    author='Amit Patel',
    author_email='amitpatel.gt@gmail.com',
    license='MIT'
    )
'''