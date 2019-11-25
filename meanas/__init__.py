"""
Electromagnetic simulation tools

See the readme or `import meanas; help(meanas)` for more info.
"""

import pathlib

from .types import dx_lists_t, field_t, vfield_t, field_updater
from .vectorization import vec, unvec

__author__ = 'Jan Petykiewicz'

with open(pathlib.Path(__file__).parent / 'VERSION', 'r') as f:
    __version__ = f.read().strip()

with open(pathlib.Path(__file__).parent.parent / 'README.md', 'r') as f:
    __doc__ = f.read()

