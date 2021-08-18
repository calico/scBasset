from pathlib import Path
from setuptools import setup, find_packages

with open('README.md') as f:
  readme = f.read()

with open('LICENSE') as f:
  license = f.read()

setup(
    name='scbasset',
    version='0.1',
    description='model scATAC with sequence-based CNN.',
    long_description=readme,
    author='Han Yuan, David Kelley',
    author_email='yuanh@calicolabs.com, drk@calicolabs.com',
    url='https://github.com/calico/scbasset',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ]
)