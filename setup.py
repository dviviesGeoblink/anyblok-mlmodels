"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
Autogenerated by poetry-setup:
https://github.com/orsinium/poetry-setup
"""
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='blok-mlmodels',  # Required
    version='0.1.0',  # Required
    description="A Blok that allows to add/update/delete/use machine learning models",  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    author="Denis Viviès",  # Optional
    author_email="legnonpi@gmail.com",  # Optional
    classifiers=['License :: OSI Approved',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'],  # Optional
    packages=find_packages(),
    install_requires=[
        'anyblok (>=0.22.5,<0.23.0)',
        'anyblok_mixins (>=1.0)',
        'psycopg2 (>=2.8,<3.0)',
    ],  # Optional
    dependency_links=[
    ],
    include_package_data=True,
    entry_points={
        'bloks': [
            'mlmodels=anyblok_mlmodels.bloks.mlmodels:MachineLearningModelBlok',
        ],
    },
    project_urls={},
)
