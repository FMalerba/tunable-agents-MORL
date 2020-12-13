from skbuild import setup
from pkg_resources import parse_requirements
import pathlib
from setuptools import find_packages

# Reading README.md to be passed to setup()
with open("../README.md", 'r') as file:
    long_description = file.read()

# Parsing requirements from requirements.txt;  these will then be passed to setup()
with pathlib.Path('../requirements.txt').open() as requirements_txt:
    install_reqs = [str(req) for req in parse_requirements(requirements_txt)]


setup(
    name='tunable_agents_environment',
    version='0.0.1',
    description="Some description",
    author='Federico Malerba',
    author_email='malerbafede@gmail.com',
    url='https://github.com/FMalerba/tunable-agents-MORL',
    long_description=long_description, 
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_reqs,
    classifiers=[" Programming Language :: Python :: 3.7", 
                 "Licence :: OSI Approved :: some_license"]
)