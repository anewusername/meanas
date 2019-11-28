"""
Electromagnetic simulation tools

See the readme or `import meanas; help(meanas)` for more info.
"""

import pathlib

__author__ = 'Jan Petykiewicz'

with open(pathlib.Path(__file__).parent / 'VERSION', 'r') as f:
    __version__ = f.read().strip()

with open(pathlib.Path(__file__).parent.parent / 'README.md', 'r') as f:
    __doc__ = f.read()

