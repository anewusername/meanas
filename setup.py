#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('meanas/VERSION', 'r') as f:
    version = f.read().strip()

setup(name='meanas',
      version=version,
      description='Electromagnetic simulation tools',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/meanas',
      packages=find_packages(),
      package_data={
          'meanas': ['VERSION']
      },
      install_requires=[
            'numpy',
            'scipy',
      ],
      extras_require={
            'test': [
                'pytest',
                'dataclasses',
                ],
            'examples': [
                'gridlock',
                ],
      },
      classifiers=[
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'Topic :: Scientific/Engineering :: Physics',
      ],
      )
