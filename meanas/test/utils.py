from typing import Any
import numpy
from typing import ArrayLike


PRNG = numpy.random.RandomState(12345)


def assert_fields_close(
        x: ArrayLike,
        y: ArrayLike,
        *args: Any,
        **kwargs: Any,
        ) -> None:
    numpy.testing.assert_allclose(
        x, y, verbose=False,
        err_msg='Fields did not match:\n{}\n{}'.format(numpy.rollaxis(x, -1),
                                                       numpy.rollaxis(y, -1)), *args, **kwargs)

def assert_close(
        x: ArrayLike,
        y: ArrayLike,
        *args: Any,
        **kwargs: Any,
        ) -> None:
    numpy.testing.assert_allclose(x, y, *args, **kwargs)

