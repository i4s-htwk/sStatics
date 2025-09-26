
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose as numpy_allclose

from sstatics.go.utils.transform import Transform


def assert_allclose(actual, desired, err_msg=''):
    numpy_allclose(actual, desired, err_msg=err_msg, atol=1e-10, rtol=1e-7)


class TestTransform(TestCase):

    def test_apply(self):
        t = Transform()
        assert_allclose(
            t.apply([0, 2], [0, -7]), np.array([[0, 2], [0, -7]]),
            err_msg='Identity transform should return the original '
                    'coordinates.'
        )
        t = Transform((1, 1), rotation=np.pi/2)
        assert_allclose(
            t.apply([0, 2], [2, 2]), np.array([[2, 2], [2, 0]]),
            err_msg='Rotation around the origin must be applied correctly.'
        )
        t = Transform((1, 1), scaling=2)
        assert_allclose(
            t.apply([0, 2], [2, 3]), np.array([[-1, 3], [3, 5]]),
            err_msg='Scaling around the origin must be applied correctly.'
        )
        t = Transform((2, 0), translation=(-1, 1))
        assert_allclose(
            t.apply([0, 2], [2, -3]), np.array([[-1, 1], [3, -2]]),
            err_msg='Translation must be applied correctly.'
        )
        t = Transform((1, 1), np.pi / 2, 2, (0, 1))
        assert_allclose(
            t.apply([0, 2], [2, 3]), np.array([[3, 5], [4, 0]]),
            err_msg='Combined transform (rotation + scaling + translation) '
                    'must be applied correctly.'
        )
        t = Transform()
        assert_allclose(
            t.apply([], []), np.array([[], []]),
            err_msg='Applying a transform to empty input must return empty '
                    'arrays.'
        )
        t = Transform(rotation=2 * np.pi)
        assert_allclose(
            t.apply([1, -1], [2, -2]), np.array([[1, -1], [2, -2]]),
            err_msg='Rotation by 2π must return the original coordinates.'
        )
        t = Transform(origin=(1, 1), scaling=0)
        assert_allclose(
            t.apply([0, 2], [2, 3]), np.array([[1, 1], [1, 1]]),
            err_msg='Scaling by zero should collapse all points to the origin.'
        )

    def test_validate(self):
        with self.assertRaises(
            TypeError, msg='Origin must be a tuple of two floats.'
        ):
            Transform(origin='not_a_tuple')
        with self.assertRaises(
            TypeError, msg='Rotation must be a float.'
        ):
            Transform(rotation='90deg')

        with self.assertRaises(
            ValueError, msg='Scaling must not be negative.'
        ):
            Transform(scaling=-1)

        with self.assertRaises(
            TypeError, msg='Translation must be a tuple of two floats.'
        ):
            Transform(translation=(1,))
