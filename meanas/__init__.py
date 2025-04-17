"""
Electromagnetic simulation tools

See the readme or `import meanas; help(meanas)` for more info.
"""

import pathlib

__version__ = '0.10'
__author__ = 'Jan Petykiewicz'


try:
    readme_path = pathlib.Path(__file__).parent / 'README.md'
    with readme_path.open('r') as f:
        __doc__ = f.read()
except Exception:
    pass

