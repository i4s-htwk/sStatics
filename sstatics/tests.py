
from itertools import product
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose as numpy_allclose

from sstatics.core import (
    Bar, BarLineLoad, BarPointLoad, BarTemp, CrossSection, Material, Node,
    NodeDisplacement, NodePointLoad
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

    # TODO: test base case
    def test_load(self):
        loads = (
            NodePointLoad(1, 1, 1), NodePointLoad(-3.5, 8, -0.2),
            NodePointLoad(3, 11, 0.65),
        )
        n = Node(0, 0, rotation=np.pi / 2, loads=loads)
        assert_allclose(n.load, np.array([[-20], [0.5], [1.45]]))
        loads = (
            NodePointLoad(1, 1, 1, rotation=0.3), NodePointLoad(-3.5, 8, -0.2),
            NodePointLoad(3, 11, 0.65, rotation=np.pi),
        )
        n = Node(0, 0, rotation=0.5, loads=loads)
        assert_allclose(
            n.load, np.array([[-3.48461279], [-4.57027778], [1.45]])
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
                if kwargs['coord'] == 'bar' and kwargs['length'] == 'proj':
                    with self.assertRaises(
                        ValueError, msg='If "coord" is set to "bar" and '
                        '"length" is set to "proj", a ValueError has to be '
                        'raised.'
                    ):
                        BarLineLoad(0, 0, **kwargs)
                    continue
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
        t1, t2, t3 = BarTemp(0, 0), BarTemp(0, 10), BarTemp(3, 10)
        assert_allclose(
            t1.temp_s, 0,
            err_msg='If no temperature is applied, the temperature change'
                    'equals to zero.')
        assert_allclose(
            t2.temp_s, 5,
            err_msg='If a temperature above or below the axis equals zero, the'
                    'temperature change is half of the entered temperature.')
        assert_allclose(t3.temp_s, 6.5)

    def test_temp_delta(self):
        t1, t2, t3 = BarTemp(0, 0), BarTemp(5, 5), BarTemp(3, 10)
        assert_allclose(t1.temp_delta, 0,
                        err_msg='If no temperature is applied, the temperature'
                                'change equals to zero.')
        assert_allclose(
            t2.temp_delta, 0,
            err_msg='If the temperature has the same value above and below the'
                    'axis, the temperature change is zero.')
        self.assertEqual(t3.temp_delta, 7)


class TestBar(TestCase):

    def test_transformation_matrix(self):
        n1 = Node(0, 0, rotation=55 * np.pi / 180)
        n2 = Node(3, -1.732050808, rotation=-55 * np.pi / 180)
        cross = CrossSection(1, 1, 1, 1, 1)
        mat = Material(1, 0.1, 1, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(b.transformation_matrix(True),
                        np.array([
                            [0.906307787036650, -0.422618261740699, 0.0,
                             0.0,
                             0.0, 0.0],
                            [0.422618261740699, 0.906307787036650, 0.0,
                             0.0,
                             0.0, 0.0],
                            [0.0, 0.0, 1.000000000000000, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.087155742747658,
                             0.996194698091746, 0.0],
                            [0.0, 0.0, 0.0, -0.996194698091746,
                             0.087155742747658, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.000000000000000]
                        ]))
        assert_allclose(b.transformation_matrix(False),
                        np.array([
                            [0.866025403784439, 0.500000000000000, 0.0,
                             0.0,
                             0.0, 0.0],
                            [-0.500000000000000, 0.866025403784439, 0.0,
                             0.0,
                             0.0, 0.0],
                            [0.0, 0.0, 1.000000000000000, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.866025403784439,
                             0.500000000000000, 0.0],
                            [0.0, 0.0, 0.0, -0.500000000000000,
                             0.866025403784439, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.000000000000000]
                        ]))

    def test_same_location(self):
        """TODO"""

    def test_inclination(self):
        n1, n2 = Node(0, 0), Node(5, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.inclination, 0,
            err_msg='If both nodes have the same z-coordinates, '
                    'the inclination equals to zero.')
        n2 = Node(0, -10)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.inclination, np.pi / 2,
            err_msg='If both nodes have the same x-coordinates and the sum of '
                    'the z-coordinates is negative, the angle is π/2.')
        n2 = Node(0, 10)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.inclination, -np.pi / 2,
            err_msg='If both nodes have the same x-coordinates and the sum of '
                    'the z-coordinates is positiv, the angle is -π/2.')
        n1, n2 = Node(-5, -10), Node(5, 10)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(b.inclination, -1.107148718)

    def test_length(self):
        n1, n2 = Node(-2, 3), Node(5, -2)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(b.length, 8.602325267)

    def test_hinge(self):
        n1, n2 = Node(0, 0), Node(1, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(b.hinge, (False, False, False, False, False,
                                  False))
        b = Bar(n1, n2, cross, mat, hinge_u_i=True, hinge_phi_j=True)
        assert_allclose(b.hinge, (True, False, False, False, False,
                                  True))

    def test_flexural_stiffness(self):
        n1, n2 = Node(0, 0), Node(1, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat, deformations=['shear'])
        assert_allclose(b.flexural_stiffness,
                        5.8149e+06)
        b = Bar(n1, n2, cross, mat, deformations=['normal', 'moment'])
        assert_allclose(b.flexural_stiffness,
                        5.8149e+03)

    def test_modified_flexural_stiffness(self):
        n1, n2 = Node(0, 0), Node(0, 5)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat, deformations=['shear', 'moment'])
        assert_allclose(b.modified_flexural_stiffness(-10),
                        5814.75112216186)

    def test_axial_rigidity(self):
        n1, n2 = Node(0, 0), Node(1, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat, deformations=['shear'])
        assert_allclose(b.axial_rigidity, 1613640000)
        b = Bar(n1, n2, cross, mat, deformations=['normal', 'moment'])
        assert_allclose(b.axial_rigidity, 1613640)

    def test_shear_stiffness(self):
        n1, n2 = Node(0, 0), Node(1, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat, deformations=['shear'])
        assert_allclose(b.shear_stiffness, 3.905819746308000e+05)
        b = Bar(n1, n2, cross, mat, deformations=['normal', 'moment'])
        assert_allclose(b.shear_stiffness, 5.814900000000001e+06)

    def test_phi(self):
        n1, n2 = Node(0, 0), Node(3, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat, deformations=['moment', 'shear'])
        assert_allclose(b.phi, 0.019850378418842)
        b = Bar(n1, n2, cross, mat, deformations=['normal'])
        assert_allclose(b.phi, 1.333333333333333)

    def test_characteristic_number(self):
        n1, n2 = Node(0, 0), Node(3, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat, deformations=['moment', 'shear'])
        assert_allclose(b.characteristic_number(-181.99971053936605),
                        0.530868169261275)

    def test_line_load(self):
        n1, n2 = Node(0, 0), Node(3, -4)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.line_load, np.array([[0], [0], [0], [0], [0], [0]]),
            err_msg='If a bar is initialized with no line loads, '
                    'then the bar line load must be a 6x1 zero vector.')
        line_loads = (BarLineLoad(1, 1, 'z', 'bar', 'exact'),
                      BarLineLoad(2, 3, 'x', 'system', 'proj'))
        b = Bar(n1, n2, cross, mat, line_loads=line_loads)
        assert_allclose(
            b.line_load,
            np.array([[0.96], [2.28], [0], [1.44], [2.92], [0]]),
            err_msg='The bar line load must equal the sum of all line loads '
                    'acting on the bar.')

    def test_point_load(self):
        n1, n2 = Node(0, 0), Node(3, -4)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.point_load, np.array([[0], [0], [0], [0], [0], [0]]),
            err_msg='If a bar is initialized with no point loads, '
                    'then the bar point load must be a 6x1 zero vector.')
        point_loads = (BarPointLoad(1, 0, 0),
                       BarPointLoad(0, 2, 3, np.pi / 4, position=1),
                       BarPointLoad(10, 30, 5, position=0.5))
        b = Bar(n1, n2, cross, mat, point_loads=point_loads)
        assert_allclose(
            b.point_load,
            np.array([[1], [0], [0], [1.414213562], [1.414213562], [3]]),
            err_msg='The bar point load must equal the sum of all '
                    'point loads acting on the bar.')

    def test_f0_point(self):
        n1, n2 = Node(0, 0), Node(2, -2)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        point_loads = (BarPointLoad(-3, 10, 0, position=0.5),
                       BarPointLoad(-3, 10, 0, position=0.333))
        b = Bar(n1, n2, cross, mat, point_loads=point_loads)
        assert_allclose(
            b.f0_point, np.array([[0], [0], [0], [0], [0], [0]]),
            err_msg='If the point loads are not acting on the beginning or '
                    'end of the bar, they are not included in the '
                    'calcultation.')
        point_loads = (BarPointLoad(-3, 10, 0, position=0),
                       BarPointLoad(5, 1, -30, position=1))
        b = Bar(n1, n2, cross, mat, point_loads=point_loads)
        assert_allclose(
            b.f0_point,
            np.array([[-9.192388155425117], [4.949747468305833], [0],
                      [2.828427124746190], [4.242640687119286], [-30]]))

    def test_f0_temp(self):
        n1, n2 = Node(0, 0), Node(2, -2)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 12 * 10 ** -6)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.f0_temp, np.array([[0], [0], [0], [0], [0], [0]]),
            err_msg='If a bar is initialized with no temp loads, '
                    'then the f0_temp must be a 6x1 zero vector.')
        b = Bar(n1, n2, cross, mat, temp=BarTemp(10, 10))
        assert_allclose(
            b.f0_temp,
            np.array([[193.6368], [0], [0], [-193.6368], [0], [0]]),
            err_msg='If the temperature is the same on both sides of the axis,'
                    'the vector only has non-zero values in the 1st and 3rd '
                    'rows.')
        b = Bar(n1, n2, cross, mat, temp=BarTemp(10, 0))
        assert_allclose(
            b.f0_temp,
            np.array([[96.818399999999997], [0], [-3.48894],
                      [-96.818399999999997], [0], [3.48894]]))

    def test_f0_displacement(self):
        displacement = NodeDisplacement(0, 0.005, 0)
        n1, n2 = Node(0, 0, displacements=displacement), Node(2, 0)
        cross = CrossSection(1943e-8, 28.48e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.f0_displacement,
            np.array([
                [0], [30.60225], [-30.60225], [0], [-30.60225],
                [-30.60225]]))
        n2 = Node(0, -5)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(
            b.f0_displacement,
            np.array(
                [[-598.08], [0], [0], [598.08], [0], [0]]))

    def test_f0_line(self):
        n1, n2 = Node(0, 0), Node(10, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = Bar(n1, n2, cross, mat)
        assert_allclose(b.f0_line,
                        np.array([[0], [0], [0], [0], [0], [0]]),
                        err_msg='If no line load is applied, the resulting'
                                'vector is a 6x1 zero vector.')
        line_load = (BarLineLoad(1, 2, 'x', 'bar', 'exact'),
                     BarLineLoad(1, 1.5, 'z', 'bar', 'exact'))
        b = Bar(n1, n2, cross, mat, line_loads=line_load)
        assert_allclose(
            b.f0_line,
            np.array([[-6.5], [-5.7500099988], [10.000049994000721],
                      [-8.5], [-6.749990001199856],
                      [-10.833283339332613]]))
        b = Bar(n1, n2, cross, mat, line_loads=line_load,
                deformations=['moment', 'normal', 'shear'])
        assert_allclose(
            b.f0_line,
            np.array([[-6.5], [-5.750148612337139], [10.000743061685700],
                      [-8.5], [-6.749851387662860],
                      [-10.832590271647636]]))
