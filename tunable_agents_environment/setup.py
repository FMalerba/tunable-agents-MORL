from skbuild import setup
from setuptools import find_packages


setup(
    name='tunable_agents_environment',
    version='0.0.1',
    description="Environment for tunable RL agents with non-linear utility functions.",
    author='Federico Malerba',
    author_email='malerbafede@gmail.com',
    url='https://github.com/FMalerba/tunable-agents-MORL',
    packages=find_packages(),
    classifiers=[" Programming Language :: Python :: 3.7"]
)