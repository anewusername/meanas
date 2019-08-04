#!/usr/bin/env python3

from setuptools import setup, find_packages
import fdfd_tools

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='fdfd_tools',
      version=fdfd_tools.version,
      description='FDFD Electromagnetic simulation tools',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/fdfd_tools',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'scipy',
      ],
      extras_require={
            'test': [
                'pytest',
                'dataclasses',
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
