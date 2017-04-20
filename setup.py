#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='fdfd_tools',
      version='0.3',
      description='FDFD Electromagnetic simulation tools',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/fdfd_tools',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'scipy',
      ],
      extras_require={
      },
      )
