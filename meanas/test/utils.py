import numpy
from numpy.typing import NDArray


PRNG = numpy.random.RandomState(12345)


def assert_fields_close(
        x: NDArray,
        y: NDArray,
        *args,
        **kwargs,
        ) -> None:
    x_disp = numpy.moveaxis(x, -1, 0)
    y_disp = numpy.moveaxis(y, -1, 0)
    numpy.testing.assert_allclose(
        x,           # type: ignore
        y,           # type: ignore
        *args,
        verbose=False,
        err_msg=f'Fields did not match:\n{x_disp}\n{y_disp}',
        **kwargs,
        )

def assert_close(
        x: NDArray,
        y: NDArray,
        *args,
        **kwargs,
        ) -> None:
    numpy.testing.assert_allclose(x, y, *args, **kwargs)

