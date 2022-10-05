"""
Electromagnetic simulation tools

See the readme or `import meanas; help(meanas)` for more info.
"""

import pathlib

__version__ = '0.8'
__author__ = 'Jan Petykiewicz'


try:
    with open(pathlib.Path(__file__).parent / 'README.md', 'r') as f:
        __doc__ = f.read()
except Exception:
    pass

