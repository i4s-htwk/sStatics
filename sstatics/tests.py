
from itertools import product
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose as numpy_allclose

from sstatics.core import (
    BarLineLoad, BarTemp, CrossSection, Node, NodeDisplacement, NodePointLoad
)


def assert_allclose(actual, desired, err_msg=''):
    numpy_allclose(actual, desired, err_msg=err_msg, atol=1e-10, rtol=1e-7)


class TestNodeDisplacement(TestCase):

    def test_vector(self):
        d = NodeDisplacement(1, 2, 0.5)
        assert_allclose(
            d.vector, np.array([[1], [2], [0.5]]),
            err_msg='The vectorized form of a node displacement must equal '
            'the 3x1 vector [[x], [z], [phi]].'
        )


class TestNodePointLoad(TestCase):

    def test_rotate(self):
        load = NodePointLoad(1, 2, 0.5)
        for rotation in (0, 2 * np.pi, 4 * np.pi):
            assert_allclose(
                load.rotate(rotation), load.vector,
                err_msg='Rotations by multiples of 2 * pi must equal the '
                'original vector.'
            )
        load = NodePointLoad(1, 2, 0.5, rotation=0.3)
        assert_allclose(
            load.rotate(0.3), load.vector,
            err_msg='Rotations by the same angle a load was instantiated with '
                    'must cancel out.'
        )
        load = NodePointLoad(1, 2, 0.5, rotation=2 * np.pi)
        assert_allclose(load.rotate(np.pi), np.array([[-1], [-2], [0.5]]))
        assert_allclose(
            load.rotate(0.3), np.array([[0.36429608], [2.20619318], [0.5]])
        )


class TestNode(TestCase):

    def test_displacement(self):
        n = Node(0, 0)
        assert_allclose(
            n.displacement, np.array([[0], [0], [0]]),
            err_msg='If a node is initialized with no displacements, then the '
            'node displacement must be a 3x1 zero vector.'
        )
        displacements = (
            NodeDisplacement(1, 1, 1), NodeDisplacement(2, 10, 0.74),
            NodeDisplacement(-4, 5, -0.3)
        )
        n = Node(0, 0, displacements=displacements)
        assert_allclose(
            n.displacement, np.array([[-1], [16], [1.44]]),
            err_msg='The node displacement must equal the sum of all '
            'displacements.'
        )

    def test_load(self):
        n = Node(0, 0)
        assert_allclose(
            n.load, np.array([[0], [0], [0]]),
            err_msg='If a node is initialized with no node point loads, then '
            'the node load must be a 3x1 zero vector.'
        )
        loads = (
            NodePointLoad(1, 1, 1), NodePointLoad(-3.5, 8, -0.2),
            NodePointLoad(3, 11, 0.65),
        )
        n = Node(0, 0, loads=loads)
        assert_allclose(
            n.load, np.array([[0.5], [20], [1.45]]),
            err_msg='The node load must equal the sum of all node point loads.'
        )

    def test_elastic_support(self):
        n = Node(0, 0)
        assert_allclose(
            n.elastic_support, np.diag([0, 0, 0]),
            err_msg='If u, w and phi of a node are initialized with keywords, '
            'then the elastic support must equal a 3x3 zero matrix.'
        )
        n = Node(0, 0, u='free', w=2.3, phi='fixed')
        assert_allclose(n.elastic_support, np.diag([0, 2.3, 0]))
        n = Node(0, 0, u=2.4, w=-0.5, phi=4)
        assert_allclose(n.elastic_support, np.diag([2.4, -0.5, 4]))

    def test_same_location(self):
        n1, n2 = Node(1, 2), Node(2, 2)
        self.assertFalse(
            n1.same_location(n2),
            'Nodes with different coordinates do not share the same location.'
        )
        n3 = Node(1, 2)
        self.assertTrue(
            n1.same_location(n3),
            'Nodes with equal coordinates share the same location.'
        )

    def test_rotate_load(self):
        loads = (
            NodePointLoad(1, 1, 1), NodePointLoad(-3.5, 8, -0.2),
            NodePointLoad(3, 11, 0.65),
        )
        n = Node(0, 0, rotation=np.pi / 2, loads=loads)
        assert_allclose(n.rotate_load(), np.array([[-20], [0.5], [1.45]]))
        loads = (
            NodePointLoad(1, 1, 1, rotation=0.3), NodePointLoad(-3.5, 8, -0.2),
            NodePointLoad(3, 11, 0.65, rotation=np.pi),
        )
        n = Node(0, 0, rotation=0.5, loads=loads)
        assert_allclose(
            n.rotate_load(), np.array([[-3.48461279], [-4.57027778], [1.45]])
        )


class TestCrossSection(TestCase):

    def test_area_validation(self):
        with self.assertRaises(
            ValueError, msg='Cross sections with an area greater than '
            'width*height must raise a ValueError.'
        ):
            CrossSection(1, 30, 5, 5, 3)


class TestBarLineLoad(TestCase):

    def test_vector(self):
        load = BarLineLoad(3.8, -5.4, direction='z')
        assert_allclose(
            load.vector, np.array([[0], [3.8], [0], [0], [-5.4], [0]]),
            err_msg='The vector of bar line loads in z-direction must equal '
            'the 6x1 vector [[0], [pi], [0], [0], [pj], [0]].'
        )
        load = BarLineLoad(3.8, -5.4, direction='x')
        assert_allclose(
            load.vector, np.array([[3.8], [0], [0], [-5.4], [0], [0]]),
            err_msg='The vector of bar line loads in z-direction must equal '
                    'the 6x1 vector [[pi], [0], [0], [pj], [0], [0]].'
        )

    def test_rotate(self):
        kwargs_combs = [
            {'direction': direction, 'coord': coord, 'length': length}
            for direction, coord, length
            in product(('x', 'z'), ('bar', 'system'), ('exact', 'proj'))
        ]
        solutions = {
            ('x', 'system', 'exact'): (
                np.array([[-3.8], [0], [0], [5.4], [0], [0]]),
                np.array([[3.63027866], [1.12297679], [0],
                          [-5.15881704], [-1.59580912], [0]]),
                np.array([[3.13627534], [-2.1456414], [0],
                          [-4.45681232], [3.04906936], [0]]),
            ),
            ('x', 'system', 'proj'): (
                np.array([[0], [0], [0], [0], [0], [0]]),
                np.array([[1.0728207], [0.33186233], [0],
                          [-1.52453468], [-0.47159384], [0]]),
                np.array([[-1.77087426], [1.21152027], [0],
                          [2.51650553], [-1.72163406], [0]]),
            ),
            ('z', 'system', 'exact'): (
                np.array([[0], [-3.8], [0], [0], [5.4], [0]]),
                np.array([[-1.12297679], [3.63027866], [0],
                          [1.59580912], [-5.15881704], [0]]),
                np.array([[2.1456414], [3.13627534], [0],
                          [-3.04906936], [-4.45681232], [0]]),
            ),
            ('z', 'system', 'proj'): (
                np.array([[0], [3.8], [0], [0], [-5.4], [0]]),
                np.array([[-1.0728207], [3.46813767], [0],
                          [1.52453468], [-4.92840616], [0]]),
                np.array([[1.77087426], [2.58847973], [0],
                          [-2.51650553], [-3.67836594], [0]]),
            ),
        }
        for kwargs in kwargs_combs:
            with self.subTest(**kwargs):
                load = BarLineLoad(3.8, -5.4, **kwargs)
                for i, rotation in enumerate((np.pi, 0.3, -0.6)):
                    if kwargs['coord'] == 'bar':
                        assert_allclose(
                            load.rotate(rotation), load.vector,
                            err_msg='If bar line loads are instantiated in '
                            'the context of the bar coordinate system, then '
                            'rotations do not have any effect.'
                        )
                    else:
                        key = (
                            kwargs['direction'], kwargs['coord'],
                            kwargs['length']
                        )
                        assert_allclose(
                            load.rotate(rotation), solutions[key][i]
                        )


class TestBarTemp(TestCase):

    def test_temp_s(self):
        t = BarTemp(3, 10)
        self.assertEqual(t.temp_s, 6.5)

    def test_temp_delta(self):
        t = BarTemp(3, 10)
        self.assertEqual(t.temp_delta, 7)
        t = BarTemp(10, 3)
        self.assertEqual(t.temp_delta, -7)
