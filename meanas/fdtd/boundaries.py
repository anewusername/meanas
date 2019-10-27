"""
Boundary conditions
"""

from typing import List, Callable, Tuple, Dict
import numpy

from .. import dx_lists_t, field_t, field_updater


def conducting_boundary(direction: int,
                        polarity: int
                        ) -> Tuple[field_updater, field_updater]:
    dirs = [0, 1, 2]
    if direction not in dirs:
        raise Exception('Invalid direction: {}'.format(direction))
    dirs.remove(direction)
    u, v = dirs

    if polarity < 0:
        boundary_slice = [slice(None)] * 3
        shifted1_slice = [slice(None)] * 3
        boundary_slice[direction] = 0
        shifted1_slice[direction] = 1

        def en(e: field_t):
            e[direction][boundary_slice] = 0
            e[u][boundary_slice] = e[u][shifted1_slice]
            e[v][boundary_slice] = e[v][shifted1_slice]
            return e

        def hn(h: field_t):
            h[direction][boundary_slice] = h[direction][shifted1_slice]
            h[u][boundary_slice] = 0
            h[v][boundary_slice] = 0
            return h

        return en, hn

    if polarity > 0:
        boundary_slice = [slice(None)] * 3
        shifted1_slice = [slice(None)] * 3
        shifted2_slice = [slice(None)] * 3
        boundary_slice[direction] = -1
        shifted1_slice[direction] = -2
        shifted2_slice[direction] = -3

        def ep(e: field_t):
            e[direction][boundary_slice] = -e[direction][shifted2_slice]
            e[direction][shifted1_slice] = 0
            e[u][boundary_slice] = e[u][shifted1_slice]
            e[v][boundary_slice] = e[v][shifted1_slice]
            return e

        def hp(h: field_t):
            h[direction][boundary_slice] = h[direction][shifted1_slice]
            h[u][boundary_slice] = -h[u][shifted2_slice]
            h[u][shifted1_slice] = 0
            h[v][boundary_slice] = -h[v][shifted2_slice]
            h[v][shifted1_slice] = 0
            return h

        return ep, hp

    raise Exception('Bad polarity: {}'.format(polarity))
