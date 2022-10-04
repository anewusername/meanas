"""
Boundary conditions

#TODO conducting boundary documentation
"""

from typing import Tuple, Any, List

from ..fdmath import fdfield_t, fdfield_updater_t


def conducting_boundary(
        direction: int,
        polarity: int
        ) -> Tuple[fdfield_updater_t, fdfield_updater_t]:
    dirs = [0, 1, 2]
    if direction not in dirs:
        raise Exception('Invalid direction: {}'.format(direction))
    dirs.remove(direction)
    u, v = dirs

    if polarity < 0:
        boundary_slice = [slice(None)] * 3      # type: List[Any]
        shifted1_slice = [slice(None)] * 3      # type: List[Any]
        boundary_slice[direction] = 0
        shifted1_slice[direction] = 1

        def en(e: fdfield_t) -> fdfield_t:
            e[direction][boundary_slice] = 0
            e[u][boundary_slice] = e[u][shifted1_slice]
            e[v][boundary_slice] = e[v][shifted1_slice]
            return e

        def hn(h: fdfield_t) -> fdfield_t:
            h[direction][boundary_slice] = h[direction][shifted1_slice]
            h[u][boundary_slice] = 0
            h[v][boundary_slice] = 0
            return h

        return en, hn

    if polarity > 0:
        boundary_slice = [slice(None)] * 3
        shifted1_slice = [slice(None)] * 3
        shifted2_slice = [slice(None)] * 3      # type: List[Any]
        boundary_slice[direction] = -1
        shifted1_slice[direction] = -2
        shifted2_slice[direction] = -3

        def ep(e: fdfield_t) -> fdfield_t:
            e[direction][boundary_slice] = -e[direction][shifted2_slice]
            e[direction][shifted1_slice] = 0
            e[u][boundary_slice] = e[u][shifted1_slice]
            e[v][boundary_slice] = e[v][shifted1_slice]
            return e

        def hp(h: fdfield_t) -> fdfield_t:
            h[direction][boundary_slice] = h[direction][shifted1_slice]
            h[u][boundary_slice] = -h[u][shifted2_slice]
            h[u][shifted1_slice] = 0
            h[v][boundary_slice] = -h[v][shifted2_slice]
            h[v][shifted1_slice] = 0
            return h

        return ep, hp

    raise Exception('Bad polarity: {}'.format(polarity))
