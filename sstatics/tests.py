
from itertools import product
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose as numpy_allclose

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.loads import (
    BarLineLoad, BarPointLoad, NodePointLoad
)
from sstatics.core.preprocessing.temperature import BarTemp
from sstatics.core.preprocessing.cross_section import CrossSection
from sstatics.core.preprocessing.material import Material
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.dof import NodeDisplacement
from sstatics.core.preprocessing.system import System
from sstatics.core.solution.solver import Solver
from sstatics.core.calc_methods.second_order import SecondOrder
from sstatics.core.preprocessing import BarSecond


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

    def test_stiffness_shear_force(self):
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        n1, n2 = Node(0, 0), Node(0, -4)
        b = Bar(n1, n2, cross, material,
                deformations=['moment', 'normal', 'shear'])
        assert_allclose(
            b.stiffness_shear_force,
            np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 0.988957461335696, 0.988957461335696, 0, 0.988957461335696,
                 0.988957461335696],
                [0, 0.988957461335696, 0.991718096001772, 0, 0.988957461335696,
                 0.983436192003544],
                [1, 0, 0, 1, 0, 0],
                [0, 0.988957461335696, 0.988957461335696, 0, 0.988957461335696,
                 0.988957461335696],
                [0, 0.988957461335696, 0.983436192003544, 0, 0.988957461335696,
                 0.991718096001772]
            ]))

    def test_f0_bar(self):

        solution = (
            np.array([[0], [-2.300024981264052], [1.600049962528104], [0],
                      [-2.699975018735948], [-1.733283370805230]]),
            np.array([[-2.300024981264052], [0], [1.600049962528104],
                      [-2.699975018735948], [0], [-1.733283370805230]]),
            np.array(
                [[-1.700006245316013], [0], [0], [-3.299993754683987], [0],
                 [-2.533308352069282]])
        )

        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        n_load = NodePointLoad(0, 182, 0, rotation=0)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        n2 = Node(0, -4, loads=n_load)

        b = Bar(n1, n2, cross, material, line_loads=line_load)

        assert_allclose(
            b.f0(True, False),
            solution[0],
            err_msg="Failed for hinge_modification = True"
        )

        assert_allclose(
            b.f0(True, True),
            solution[1],
            err_msg="Failed for hinge_modification and to_node_coord = True"
        )

        b = Bar(n1, n2, cross, material, line_loads=line_load,
                hinge_phi_i=True)

        assert_allclose(
            b.f0(True, True),
            solution[2],
            err_msg="Failed for modified Bar and "
                    " (hinge_phi_i=True)"
        )

    def test_stiffness_matrix(self):
        solutions = (
            np.array([
                [1090.29375, 0, -2180.5875, -1090.29375, 0, -2180.5875],
                [0, 403410, 0, 0, -403410, 0],
                [-2180.5875, 0, 5814.9, 2180.5875, 0, 2907.45],
                [-1090.29375, 0, 2180.5875, 1090.29375, 0, 2180.5875],
                [0, -403410, 0, 0, 403410, 0],
                [-2180.5875, 0, 2907.45, 2180.5875, 0, 5814.9]
            ]),
            np.array([
                [1078.25413911018, 0, -2156.50827822035, -1078.25413911018, 0,
                 -2156.50827822035],
                [0, 403410, 0, 0, -403410, 0],
                [-2156.50827822035, 0, 5766.74155644071, 2156.50827822035, 0,
                 2859.29155644071],
                [-1078.25413911018, 0, 2156.50827822035, 1078.25413911018, 0,
                 2156.50827822035],
                [0, -403410, 0, 0, 403410, 0],
                [-2156.50827822035, 0, 2859.29155644071, 2156.50827822035, 0,
                 5766.74155644071]
            ])
        )

        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        n_load = NodePointLoad(0, 182, 0, rotation=0)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        n2 = Node(0, -4, loads=n_load)
        b = Bar(n1, n2, cross, material, line_loads=line_load,
                deformations=['moment', 'normal'])
        assert_allclose(
            b.stiffness_matrix(True, True),
            solutions[0],
            err_msg="Failed for hinge_modification and to_node_coord = True "
        )
        b = Bar(n1, n2, cross, material, line_loads=line_load,
                deformations=['moment', 'normal', 'shear'])
        assert_allclose(
            b.stiffness_matrix(True, True),
            solutions[1],
            err_msg="Failed for shear deformation and hinge_modification and"
                    "to_node_coord = True"
        )


class TestBarSecond(TestCase):

    def test_modified_flexural_stiffness(self):
        n1, n2 = Node(0, 0), Node(0, 5)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = BarSecond(n1, n2, cross, mat, deformations=['shear', 'moment'],
                      f_axial=-10)
        assert_allclose(b.modified_flexural_stiffness,
                        5814.75112216186)

    def test_characteristic_number(self):
        n1, n2 = Node(0, 0), Node(3, 0)
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        mat = Material(210000000, 0.1, 81000000, 0.1)
        b = BarSecond(n1, n2, cross, mat, deformations=['moment', 'shear'],
                      f_axial=-181.99971053936605)
        assert_allclose(b.characteristic_number,
                        0.530868169261275)

    def test_f0_line(self):
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1, n2 = Node(0, 0), Node(0, -4)
        b_analytic = BarSecond(
            n1, n2, cross, material, line_loads=line_load,
            approach='analytic', f_axial=-181.99971053936605)
        b_taylor = BarSecond(
            n1, n2, cross, material, line_loads=line_load,
            approach='taylor', f_axial=-181.99971053936605)
        b_p_delta = BarSecond(
            n1, n2, cross, material, line_loads=line_load,
            approach='p_delta', f_axial=-181.99971053936605)
        b_p_delta_first = Bar(
            n1, n2, cross, material, line_loads=line_load)
        assert_allclose(b_analytic.f0_line, b_analytic.f0_line_analytic)
        assert_allclose(b_taylor.f0_line, b_taylor.f0_line_taylor)
        assert_allclose(b_p_delta.f0_line, b_p_delta_first.f0_line)

    def test_f0_line_analytic(self):
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1, n2 = Node(0, 0), Node(0, -4)
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=-181.99971053936605)
        assert_allclose(
            b.f0_line_analytic,
            np.array([[0], [-2.299903905345166], [1.613939467730169],
                      [0], [-2.700096094654834], [-1.747657179682811]]))
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=181.99971053936605)
        assert_allclose(
            b.f0_line_analytic,
            np.array([[0], [-2.300144713318061], [1.586491286791585],
                      [0], [-2.699855286681939], [-1.719245766852758]]))

    def test_f0_line_taylor(self):
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1, n2 = Node(0, 0), Node(0, -4)
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=-181.99971053936605)
        assert_allclose(
            b.f0_line_taylor,
            np.array([[0], [-2.300024975792029], [1.600049990707627],
                      [0], [-2.699975024207971], [-1.733283420872844]]))
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=181.99971053936605)
        assert_allclose(
            b.f0_line_taylor,
            np.array([[0], [-2.300024986736074], [1.600049934348581],
                      [0], [-2.699975013263926], [-1.733283320737616]]))

    def test_stiffness_second_order_analytic(self):
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1, n2 = Node(0, 0), Node(0, -4)
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=-181.99971053936605)
        assert_allclose(
            b.stiffness_matrix_analytic,
            np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 0.949154629596613, 0.990855423302253, 0, 0.949154629596613,
                 0.990855423302253],
                [0, 0.990855423302253, 0.982612659783107, 0, 0.990855423302253,
                 1.007340950340543],
                [1, 0, 0, 1, 0, 0],
                [0, 0.949154629596613, 0.990855423302253, 0, 0.949154629596613,
                 0.990855423302253],
                [0, 0.990855423302253, 1.007340950340543, 0, 0.990855423302253,
                 0.982612659783107]]))
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=181.99971053936605)
        assert_allclose(
            b.stiffness_matrix_analytic,
            np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 1.049286243245867, 1.007585973187277, 0, 1.049286243245867,
                 1.007585973187277],
                [0, 1.007585973187277, 1.016044207086668, 0, 1.007585973187277,
                 0.990669505388496],
                [1, 0, 0, 1, 0, 0],
                [0, 1.049286243245867, 1.007585973187277, 0, 1.049286243245867,
                 1.007585973187277],
                [0, 1.007585973187277, 0.990669505388496, 0, 1.007585973187277,
                 1.016044207086668]
            ]))

    def test_stiffness_second_order_taylor(self):
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1, n2 = Node(0, 0), Node(0, -4)
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=-181.99971053936605)
        assert_allclose(
            b.stiffness_matrix_taylor,
            np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 0.949184922004305, 0.990885453929866, 0, 0.949184922004305,
                 0.990885453929866],
                [0, 0.990885453929866, 0.982723314147304, 0, 0.990885453929866,
                 1.007209733494989],
                [1, 0, 0, 1, 0, 0],
                [0, 0.949184922004305, 0.990885453929866, 0, 0.949184922004305,
                 0.990885453929866],
                [0, 0.990885453929866, 1.007209733494989, 0, 0.990885453929866,
                 0.982723314147304]]))
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=181.99971053936605)
        assert_allclose(
            b.stiffness_matrix_taylor,
            np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 1.049316199412877, 1.007615669443616, 0, 1.049316199412877,
                 1.007615669443616],
                [0, 1.007615669443616, 1.016152528382807, 0, 1.007615669443616,
                 0.990541951565234],
                [1, 0, 0, 1, 0, 0],
                [0, 1.049316199412877, 1.007615669443616, 0, 1.049316199412877,
                 1.007615669443616],
                [0, 1.007615669443616, 0.990541951565234, 0, 1.007615669443616,
                 1.016152528382807]
            ]))

    def test_stiffness_second_order_p_delta(self):
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1, n2 = Node(0, 0), Node(0, -4)
        b = BarSecond(n1, n2, cross, material, line_loads=line_load,
                      f_axial=-181.99971053936605)
        assert_allclose(
            b.stiffness_matrix_p_delta,
            np.array([
                [0, 0, 0, 0, 0, 0],
                [0, -45.499927634841512, 0, 0, 45.499927634841512, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 45.499927634841512, 0, 0, -45.499927634841512, 0],
                [0, 0, 0, 0, 0, 0]]))

    def test_f0_bar(self):
        f_axial = -181.99971053936605

        approaches = ['analytic', 'taylor', 'p_delta']

        solutions = {
            ('analytic'): (
                np.array([[0], [-2.299903905345166], [1.613939467730169], [0],
                          [-2.700096094654834], [-1.747657179682811]]),
                np.array([[-2.299903905345166], [0], [1.613939467730169],
                          [-2.700096094654834], [0], [-1.747657179682811]])
            ),
            ('taylor'): (
                np.array([[0], [-2.300024975792029], [1.600049990707627], [0],
                          [-2.699975024207971], [-1.733283420872844]]),
                np.array([[-2.300024975792029], [0], [1.600049990707627],
                          [-2.699975024207971], [0], [-1.733283420872844]])
            ),
            ('p_delta'): (
                np.array([[0], [-2.300024981264052], [1.600049962528104], [0],
                          [-2.699975018735948], [-1.733283370805230]]),
                np.array([[-2.300024981264052], [0], [1.600049962528104],
                          [-2.699975018735948], [0], [-1.733283370805230]])
            )
        }
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        n_load = NodePointLoad(0, 182, 0, rotation=0)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        n2 = Node(0, -4, loads=n_load)

        for approach in approaches:
            bar = BarSecond(
                node_i=n1,
                node_j=n2,
                cross_section=cross,
                material=material,
                line_loads=line_load,
                approach=approach,
                f_axial=f_axial
            )
            result1 = bar.f0(True, False)
            result2 = bar.f0(True, True)
            ref1, ref2 = solutions[approach]

            assert_allclose(
                result1, ref1,
                err_msg=f"Failed f0() test 1 for approach={approach}"
            )

            assert_allclose(
                result2, ref2,
                err_msg=f"Failed f0() test 2 for approach={approach}"
            )

            bar_hinge = BarSecond(
                node_i=n1,
                node_j=n2,
                cross_section=cross,
                material=material,
                line_loads=line_load,
                approach='p_delta',
                f_axial=f_axial,
                hinge_phi_i=True
            )

            result_hinge = bar_hinge.f0(True, True)
            ref_hinge = np.array(
                [[-1.700006245316013], [0], [0],
                 [-3.299993754683987], [0], [-2.533308352069282]]
            )

            assert_allclose(
                result_hinge, ref_hinge,
                err_msg="Failed for BarSecond with hinge_phi_i=True"
            )

    def test_stiffness_matrix(self):
        f_axial = -181.99971053936605
        approaches = ['analytic', 'taylor', 'p_delta']

        solutions = {
            ('analytic'): (
                np.array([
                    [1034.85736043275, 0, -2160.6469503601, -1034.85736043275,
                     0, -2160.6469503601],
                    [0, 403410, 0, 0, -403410, 0],
                    [-2160.6469503601, 0, 5713.79435537279, 2160.6469503601, 0,
                     2928.79344606761],
                    [-1034.85736043275, 0, 2160.6469503601, 1034.85736043275,
                     0, 2160.6469503601],
                    [0, -403410, 0, 0, 403410, 0],
                    [-2160.6469503601, 0, 2928.79344606761, 2160.6469503601, 0,
                     5713.79435537279]
                ]),
                np.array([
                    [1023.82229444851, 0, -2137.64836291218, -1023.82229444851,
                     0, -2137.64836291218],
                    [0, 403410, 0, 0, -403410, 0],
                    [-2137.64836291218, 0, 5667.1650617622, 2137.64836291218,
                     0, 2883.42838988652],
                    [-1023.82229444851, 0, 2137.64836291218, 1023.82229444851,
                     0, 2137.64836291218],
                    [0, -403410, 0, 0, 403410, 0],
                    [-2137.64836291218, 0, 2883.42838988652, 2137.64836291218,
                     0, 5667.1650617622]
                ])
            ),
            ('taylor'): (
                np.array([
                    [1034.89038805553, 0, -2160.71243477129, -1034.89038805553,
                     0, -2160.71243477129],
                    [0, 403410, 0, 0, -403410, 0],
                    [-2160.71243477129, 0, 5714.43779943516, 2160.71243477129,
                     0, 2928.41193965001],
                    [-1034.89038805553, 0, 2160.71243477129, 1034.89038805553,
                     0, 2160.71243477129],
                    [0, -403410, 0, 0, 403410, 0],
                    [-2160.71243477129, 0, 2928.41193965001, 2160.71243477129,
                     0, 5714.43779943516]
                ]),
                np.array([
                    [1023.85931267192, 0, -2137.71407426467, -1023.85931267192,
                     0, -2137.71407426467],
                    [0, 403410, 0, 0, -403410, 0],
                    [-2137.71407426467, 0, 5667.80918526337, 2137.71407426467,
                     0, 2883.04711179532],
                    [-1023.85931267192, 0, 2137.71407426467, 1023.85931267192,
                     0, 2137.71407426467],
                    [0, -403410, 0, 0, 403410, 0],
                    [-2137.71407426467, 0, 2883.04711179532, 2137.71407426467,
                     0, 5667.80918526337]
                ])
            ),
            ('p_delta'): (
                np.array([
                    [1044.79382236516, 0, -2180.5875, -1044.79382236516, 0,
                     -2180.5875],
                    [0, 403410, 0, 0, -403410, 0],
                    [-2180.5875, 0, 5814.9, 2180.5875, 0, 2907.45],
                    [-1044.79382236516, 0, 2180.5875, 1044.79382236516, 0,
                     2180.5875],
                    [0, -403410, 0, 0, 403410, 0],
                    [-2180.5875, 0, 2907.45, 2180.5875, 0, 5814.9]
                ]),
                np.array([
                    [1032.75421147533, 0, -2156.50827822035, -1032.75421147533,
                     0, -2156.50827822035],
                    [0, 403410, 0, 0, -403410, 0],
                    [-2156.50827822035, 0, 5766.74155644071, 2156.50827822035,
                     0, 2859.29155644071],
                    [-1032.75421147533, 0, 2156.50827822035, 1032.75421147533,
                     0, 2156.50827822035],
                    [0, -403410, 0, 0, 403410, 0],
                    [-2156.50827822035, 0, 2859.29155644071, 2156.50827822035,
                     0, 5766.74155644071]
                ]))
        }
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        n_load = NodePointLoad(0, 182, 0, rotation=0)
        line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        n2 = Node(0, -4, loads=n_load)

        for approach in approaches:
            bar_1 = BarSecond(
                node_i=n1,
                node_j=n2,
                cross_section=cross,
                material=material,
                line_loads=line_load,
                approach=approach,
                f_axial=f_axial
            )
            bar_2 = BarSecond(
                node_i=n1,
                node_j=n2,
                cross_section=cross,
                material=material,
                line_loads=line_load,
                deformations=['moment', 'normal', 'shear'],
                approach=approach,
                f_axial=f_axial
            )
            result1 = bar_1.stiffness_matrix(True, True)
            result2 = bar_2.stiffness_matrix(True, True)
            ref1, ref2 = solutions[approach]

            assert_allclose(
                result1, ref1,
                err_msg=f"Failed stiffness_matrix() test 1 for "
                        f"approach={approach}"
            )

            assert_allclose(
                result2, ref2,
                err_msg=f"Failed stiffness_matrix() test shear for "
                        f"approach={approach}"
            )


class TestSystem(TestCase):

    def test_connected_nodes(self):
        """TODO"""

    def test_node_to_bar_map(self):
        """TODO"""

    def test_nodes(self):
        """TODO"""

    def test_get_polplan(self):
        """TODO"""


class TestSolver(TestCase):

    def test_get_zero_matrix(self):
        # Check that _get_zero_matrix returns correct zero-size for system DOFs
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        n1, n2, n3, n4 = Node(0, 0), Node(0, 2), Node(0, 4), Node(6, 0)
        b1 = Bar(n1, n2, cross, material)
        b2 = Bar(n2, n3, cross, material)
        b3 = Bar(n3, n4, cross, material)
        system1 = System([b1])
        system2 = System([b1, b2, b3])
        assert_allclose(
            Solver(system1)._get_zero_matrix(),
            np.zeros((6, 6)),
            err_msg='If a system has one bar without it being segmented, the '
                    'matrix must be a 6x6 zero matrix.')
        assert_allclose(
            Solver(system2)._get_zero_matrix(),
            np.zeros((12, 12)),
            'If a system has three bars without it being segmented, the '
            'matrix must be a 12x12 zero matrix.'
        )

    def test_get_zero_vec(self):
        # Check that _get_zero_vec returns correct zero-size for system DOFs
        cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        material = Material(210000000, 0.1, 81000000, 0.1)
        n1, n2, n3, n4 = Node(0, 0), Node(0, 2), Node(0, 4), Node(6, 0)
        b1 = Bar(n1, n2, cross, material)
        b2 = Bar(n2, n3, cross, material)
        b3 = Bar(n3, n4, cross, material)
        system1 = System([b1])
        system2 = System([b1, b2, b3])
        assert_allclose(
            Solver(system1)._get_zero_vec(),
            np.zeros((6, 1)),
            err_msg='If a system has one bar without it being segmented, the '
                    'vector must be a 6x1 zero vector.')
        assert_allclose(
            Solver(system2)._get_zero_vec(),
            np.zeros((12, 1)),
            err_msg='If a system has three bars without it being segmented, '
                    'the vector must be a 12x1 zero vector.')

    def test_stiffness_matrix(self):
        # Check stiffness matrix content, symmetry, and caching
        material = Material(210000000, 0.1, 81000000, 0.1)
        cross1 = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        cross2 = CrossSection(349e-8, 21.2e-4, 0.096, 0.1, 0.1)
        n1, n2, n3, n4 = Node(0, 0), Node(4, 0), Node(6, 0), Node(4, 4)
        b1 = Bar(n1, n2, cross1, material)
        b2 = Bar(n2, n3, cross2, material)
        b3 = Bar(n2, n4, cross1, material)
        system1 = System([b1, b2])
        system2 = System([b1, b2, b3])
        assert_allclose(
            Solver(system1).stiffness_matrix,
            np.array([
                [149625, 0, 0, -149625, 0, 0, 0, 0, 0],
                [0, 763.875, -1527.75, 0, -763.875, -1527.75, 0, 0, 0],
                [0, -1527.75, 4074, 0, 1527.75, 2037, 0, 0, 0],
                [-149625, 0, 0, 372225, 0, 0, -222600, 0, 0],
                [0, -763.875, 1527.75, 0, 1863.225, 428.4, 0, -1099.35,
                 -1099.35],
                [0, -1527.75, 2037, 0, 428.4, 5539.8, 0, 1099.35, 732.9],
                [0, 0, 0, -222600, 0, 0, 222600, 0, 0],
                [0, 0, 0, 0, -1099.35, 1099.35, 0, 1099.35, 1099.35],
                [0, 0, 0, 0, -1099.35, 732.9, 0, 1099.35, 1465.8]]),
            err_msg='Unexpected stiffness matrix values for system1.')
        assert_allclose(
            Solver(system2).stiffness_matrix,
            np.array([
                [149625, 0, 0, -149625, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 763.875, -1527.75, 0, -763.875, -1527.75, 0, 0, 0, 0, 0,
                 0],
                [0, -1527.75, 4074, 0, 1527.75, 2037, 0, 0, 0, 0, 0, 0],
                [-149625, 0, 0, 372988.875, 0, 1527.75, -222600, 0, 0,
                 -763.875, 0, 1527.75],
                [0, -763.875, 1527.75, 0, 151488.225, 428.4, 0, -1099.35,
                 -1099.35, 0, -149625, 0],
                [0, -1527.75, 2037, 1527.75, 428.4, 9613.8, 0, 1099.35, 732.9,
                 -1527.75, 0, 2037],
                [0, 0, 0, -222600, 0, 0, 222600, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1099.35, 1099.35, 0, 1099.35, 1099.35, 0, 0, 0],
                [0, 0, 0, 0, -1099.35, 732.9, 0, 1099.35, 1465.8, 0, 0, 0],
                [0, 0, 0, -763.875, 0, -1527.75, 0, 0, 0, 763.875, 0,
                 -1527.75],
                [0, 0, 0, 0, -149625, 0, 0, 0, 0, 0, 149625, 0],
                [0, 0, 0, 1527.75, 0, 2037, 0, 0, 0, -1527.75, 0, 4074]]),
            err_msg='Unexpected stiffness matrix values for system1.')
        # Check matrix symmetry
        k1 = Solver(system1).stiffness_matrix
        assert_allclose(k1, k1.T,
                        err_msg="Stiffness matrix must be symmetric.")
        # Check caching of stiffness_matrix
        fo_sym = Solver(system1)
        m1 = fo_sym.stiffness_matrix
        m2 = fo_sym.stiffness_matrix
        self.assertIs(m1, m2,
                      "stiffness_matrix should be cached on instance.")

    def test_elastic_matrix(self):
        # Check elastic matrix assembly and that off-diagonal terms are zero
        material = Material(210000000, 0.1, 81000000, 0.1)
        cross1 = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        n1, n2 = Node(0, 0), Node(4, 0)
        b1 = Bar(n1, n2, cross1, material)
        system1 = System([b1])
        assert_allclose(
            Solver(system1).elastic_matrix,
            np.zeros((6, 6)),
            err_msg='If nodes are not elastically supported, a zero elastic '
                    'matrix must be returned.')
        n3 = Node(4, 0, u=100, phi=1000)
        b1 = Bar(n1, n3, cross1, material)
        system1 = System([b1])
        assert_allclose(Solver(system1).elastic_matrix,
                        np.array([
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 100, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1000]]),
                        err_msg='Elastic support values must appear on the '
                                'diagonal blocks only.')
        # Check off-diagonal of elastic matrix equals zero
        em = Solver(system1).elastic_matrix
        off = em - np.diag(np.diag(em))
        assert_allclose(
            off, np.zeros_like(off),
            err_msg="Elastic matrix must be strictly diagonal-blocked "
                    "(no off-diagonal terms)."
        )

    def test_system_matrix(self):
        # Check system_matrix equals stiffness + elastic and is cached
        material = Material(210000000, 0.1, 81000000, 0.1)
        cross1 = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        cross2 = CrossSection(349e-8, 21.2e-4, 0.096, 0.1, 0.1)
        n1, n2, n3 = Node(0, 0, phi=800), Node(4, 0), Node(6, 0, u=100)
        b1 = Bar(n1, n2, cross1, material)
        b2 = Bar(n2, n3, cross2, material)
        system1 = System([b1, b2])
        # Check expected numerical values (golden matrix)
        assert_allclose(
            Solver(system1).system_matrix,
            np.array([
                [149625, 0, 0, -149625, 0, 0, 0, 0, 0],
                [0, 763.875, -1527.75, 0, -763.875, -1527.75, 0, 0, 0],
                [0, -1527.75, 4874, 0, 1527.75, 2037, 0, 0, 0],
                [-149625, 0, 0, 372225, 0, 0, -222600, 0, 0],
                [0, -763.875, 1527.75, 0, 1863.225, 428.4, 0, -1099.35,
                 -1099.35],
                [0, -1527.75, 2037, 0, 428.4, 5539.8, 0, 1099.35, 732.9],
                [0, 0, 0, -222600, 0, 0, 222700, 0, 0],
                [0, 0, 0, 0, -1099.35, 1099.35, 0, 1099.35, 1099.35],
                [0, 0, 0, 0, -1099.35, 732.9, 0, 1099.35, 1465.8]]),
            err_msg='system_matrix must equal '
                    'stiffness_matrix + elastic_matrix.')
        # Check caching (same object instance)
        fo_sys = Solver(system1)
        s1 = fo_sys.system_matrix
        s2 = fo_sys.system_matrix
        self.assertIs(s1, s2, "system_matrix should be cached on instance.")

    def test_f0_1_order(self):
        material = Material(210000000, 0.1, 81000000, 0.1)
        cross1 = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        n1, n2 = Node(0, 0), Node(4, 0)
        n3, n4 = Node(4, 4, u='fixed', w='fixed', phi='fixed'), Node(6, 0)
        l_load1, l_load2 = BarLineLoad(1, 1), BarLineLoad(2, 2)
        l_load3 = BarLineLoad(3, 4)
        b1 = Bar(n1, n2, cross1, material, line_loads=l_load1)
        b2 = Bar(n2, n3, cross1, material, line_loads=l_load2)
        b3 = Bar(n2, n4, cross1, material, line_loads=l_load3)
        # Check expected f0 values for multiple bars
        assert_allclose(
            Solver(System([b1, b2, b3])).f0,
            np.array([[0], [-2], [1.3333333333], [4], [-5.3000997009],
                      [2.4667663676], [4], [0], [-2.6666666667],
                      [0], [-3.6999002991], [-1.1999002991]]),
            err_msg='Unexpected f0 vector for the given line loads.')
        # Check zero f0 if no loads exist
        b1 = Bar(n1, n2, cross1, material)
        b2 = Bar(n2, n3, cross1, material)
        b3 = Bar(n2, n4, cross1, material)
        assert_allclose(
            Solver(System([b1, b2, b3])).f0,
            np.zeros((12, 1)),
            err_msg='If the system is not subjected to loads, a zero vector is'
                    'returned.')
        # Check invariance when bar is segmented vs. not segmented (same f0)
        n_load1, n_load2 = NodePointLoad(1, 20, 0), NodePointLoad(5, 0, 3)
        n1, n2 = Node(0, 0, loads=[n_load1, n_load2]), Node(4, 0,
                                                            loads=n_load2)
        b_load = BarPointLoad(10, 0, 0, 0, 0.5)
        bar1 = Bar(n1, n2, cross1, material, line_loads=l_load1,
                   point_loads=b_load)
        n5 = Node(2, 0)
        bar2 = Bar(n1, n5, cross1, material, line_loads=l_load1)
        bar3 = Bar(n5, n2, cross1, material, line_loads=l_load1)
        assert_allclose(
            Solver(System([bar1])).f0,
            Solver(System([bar2, bar3])).f0,
            err_msg='f0 must be identical whether the bar is segmented '
                    'by helper or manually.')

    def test_p0(self):
        # Check p0 assembly from node and bar point loads
        material = Material(210000000, 0.1, 81000000, 0.1)
        cross1 = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        n_load1, n_load2 = NodePointLoad(1, 20, 0), NodePointLoad(5, 0, 3)
        b_load1 = BarPointLoad(-50, 0, 10, position=0.5)
        n1 = Node(0, 0, loads=[n_load1, n_load2])
        n2 = Node(4, 0, loads=n_load2)
        l_load1 = BarLineLoad(1, 1)
        b1 = Bar(n1, n2, cross1, material, line_loads=l_load1,
                 point_loads=b_load1)
        # Check p0 collects node and bar point loads
        assert_allclose(
            Solver(System([b1])).p0,
            np.array([[6], [20], [3], [-50], [0], [10], [5], [0], [3]]),
            err_msg='If a BarPointLoad is applied to a bar at a position '
                    'between 0 and 1, the load is assigned to the p0 vector.')
        # Check p0 without bar point load
        b1 = Bar(n1, n2, cross1, material)
        assert_allclose(
            Solver(System([b1])).p0,
            np.array([[6], [20], [3], [5], [0], [3]]),
            err_msg='p0 must equal the sum of nodal loads when no bar '
                    'point load is present.')

    def test_p(self):
        # Check p identity p = p0 - f0
        material = Material(210000000, 0.1, 81000000, 0.1)
        cross1 = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        n_load1, n_load2 = NodePointLoad(1, 20, 0), NodePointLoad(5, 0, 3)
        b_load1 = BarPointLoad(-50, 0, 10, position=0.5)
        n1 = Node(0, 0, loads=[n_load1, n_load2])
        n2 = Node(4, 0, loads=n_load2)
        l_load1 = BarLineLoad(1, 1)
        b1 = Bar(n1, n2, cross1, material, line_loads=l_load1,
                 point_loads=b_load1)
        # Check numeric expected p
        assert_allclose(
            Solver(System([b1])).p,
            np.array([[6], [21], [2.6666666667], [-50], [2],
                      [10], [5], [1], [3.3333333]]),
            err_msg='Unexpected p vector for the given load combination.')
        # Check identity p = p0 - f0
        fo = Solver(System([b1]))
        assert_allclose(fo.p, fo.p0 - fo.f0,
                        err_msg="Global load vector must satisfy p = p0 - f0.")

    def test_boundary_conditions(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1 = Node(0, 0, u='fixed', w='fixed')

        # Use fixed coordinates
        n2 = Node(7.0, 2.0, w='fixed')
        load = BarLineLoad(1, 1)
        b1 = Bar(n1, n2, cross, mat, line_loads=load)
        fo_bc = Solver(System([b1]))

        # Check diagonal entries after BCs are non-zero
        diagonal = np.diag(fo_bc.boundary_conditions[0])
        self.assertTrue(
            np.all(diagonal != 0),
            'All diagonal entries must be > 0 after applying '
            'boundary conditions.'
        )

        # Check K_mod d = p_mod
        k_mod, p_mod = fo_bc.boundary_conditions
        d = fo_bc.node_deform
        assert_allclose(
            k_mod @ d, p_mod,
            err_msg="Modified linear system must satisfy K_mod * d = p_mod."
        )

        # Check that boundary_conditions does not mutate system_matrix or p
        k_before = fo_bc.system_matrix.copy()
        p_before = fo_bc.p.copy()
        _ = fo_bc.boundary_conditions
        assert_allclose(
            fo_bc.system_matrix, k_before,
            err_msg="boundary_conditions must not mutate system_matrix."
        )
        assert_allclose(
            fo_bc.p, p_before,
            err_msg="boundary_conditions must not mutate p."
        )

        # Check known modification case
        n2 = Node(10, 0, w='fixed')
        system2 = System([Bar(n1, n2, cross, mat, line_loads=load)])
        assert_allclose(
            Solver(system2).boundary_conditions[0],
            np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1629.6, 0, 0, 814.8],
                      [0, 0, 0, 59850, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 814.8, 0, 0, 1629.6]]),
            err_msg="Unexpected modified stiffness matrix for the "
                    "given supports."
        )
        assert_allclose(
            Solver(system2).boundary_conditions[1],
            np.array([[0], [0], [-8.3333333], [0], [0], [8.3333333]]),
            err_msg="Unexpected modified load vector for the given supports."
        )

        # Check no modifications if no supports defined
        n1, n2 = Node(0, 0), Node(4, 0)
        b1 = Bar(n1, n2, cross, mat)
        system2 = System([b1])
        assert_allclose(
            Solver(system2).system_matrix,
            Solver(system2).boundary_conditions[0],
            err_msg='No boundary modifications expected when no supports '
                    'are defined (k unchanged).'
        )
        assert_allclose(
            Solver(system2).p,
            Solver(system2).boundary_conditions[1],
            err_msg='No boundary modifications expected when no supports '
                    'are defined (p unchanged).'
        )

        # Check support reaction transformation: rotation == 0 -> identical
        n1r = Node(0, 0, u='fixed', w='fixed', phi='fixed', rotation=0)
        n2r = Node(10, 0, w='fixed')
        fo_rot0 = Solver(
            System([Bar(n1r, n2r, cross, mat, line_loads=load)]))
        assert_allclose(
            fo_rot0.system_support_forces,
            fo_rot0.node_support_forces,
            err_msg="If rotation=0, system_support_forces must match "
                    "node_support_forces exactly."
        )

        # Check support reaction transformation: rotation != 0
        n1r = Node(0, 0, u='fixed', w='fixed', phi='fixed',
                   rotation=np.pi / 2)
        fo_rot = Solver(
            System([Bar(n1r, n2r, cross, mat, line_loads=load)]))
        local = fo_rot.node_support_forces[:3, :]
        globl = fo_rot.system_support_forces[:3, :]
        self.assertFalse(
            np.allclose(local, globl),
            "With node rotation, transformed support forces must differ "
            "from local ones."
        )
        assert_allclose(
            np.linalg.norm(local[:2, :]),
            np.linalg.norm(globl[:2, :]),
            err_msg="Rotation should change force components but preserve "
                    "the norm in the u,w-plane."
        )

    def test_node_deform(self):
        # Check node_deform shape and that K_mod d = p_mod holds
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1, n2 = Node(0, 0, u='fixed', w='fixed'), Node(3, 0, w='fixed')
        b1 = Bar(n1, n2, cross, mat,
                 point_loads=BarPointLoad(0, 10, 0, position=0.5))
        system = System([b1])

        n2b = Node(10, 0, w='fixed')
        load = BarLineLoad(1, 1)
        b1b = Bar(n1, n2b, cross, mat, line_loads=load)
        system1 = System([b1b])

        # Check shape equals dof * number_of_nodes
        deform = Solver(system).node_deform
        expected_shape = (Solver(system).dof * len(system.nodes()), 1)
        self.assertEqual(
            deform.shape, expected_shape,
            'The deformation vector length must be dof * number of nodes.'
        )

        # Check expected values for system1
        assert_allclose(
            Solver(system1).node_deform,
            np.array([[0], [0], [-0.010227458681026], [0], [0],
                      [0.010227458681026]]),
            err_msg='Unexpected nodal deformation values for the '
                    'given load case.'
        )

        # Check K_mod d = p_mod for system1
        fo_nd = Solver(system1)
        k_mod, p_mod = fo_nd.boundary_conditions
        d = fo_nd.node_deform
        assert_allclose(
            k_mod @ d, p_mod,
            err_msg="Nodal deformation must satisfy the modified "
                    "linear system."
        )

    def test_node_deform_list(self):
        # Check node_deform_list construction for multi-bar systems
        cross1 = CrossSection(0.3 * 0.5 ** 3/12, 0.3 * 0.5, 0.5, 0.3, 0.1)
        cross2 = CrossSection(0.3 * 0.3 ** 3 / 12, 0.3 * 0.3, 0.3, 0.3, 0.1)
        mat1 = Material(3000e3, 1.2, 60e3, 0.1)
        n1, n2 = Node(0, 0, w='fixed'), Node(2, 0)
        n3 = Node(5, 0, w='fixed')
        n4 = Node(2, 2.5, u='fixed', w='fixed', phi='fixed')
        b1 = Bar(n1, n2, cross1, mat1, line_loads=BarLineLoad(30, 30),
                 deformations=['moment'])
        b2 = Bar(n2, n3, cross1, mat1, deformations=['moment'])
        b3 = Bar(n4, n2, cross2, mat1,
                 point_loads=BarPointLoad(60, 0, 0, 0, 0.6),
                 deformations=['moment'])
        system = System([b1, b2, b3])
        assert_allclose(
            Solver(system).node_deform_list,
            [np.array([[0.025618694], [0], [-0.0002861296],
                       [0.025618694], [0.0000003651], [-0.0004949552]]),
             np.array([[0.025618694], [0.0000003651], [-0.0004949552],
                       [0.025618694], [0], [0.0002476602]]),
             np.array([[0], [0], [0], [0.0185560632], [0.0000002191],
                       [-0.0136303065]]),
             np.array([[0.0185560632], [0.0000002191], [-0.0136303065],
                       [0.025618694], [0.0000003651], [-0.0004949552]])],
            err_msg='Unexpected node_deform_list for the given multi-bar '
                    'system.')

    def test_bar_deform(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1, n2 = (Node(0, 0, u='fixed', w='fixed', phi='fixed'),
                  Node(3, 0, w='fixed'))
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        fo = Solver(System([b1]))
        bd = fo.bar_deform[0]
        # Check definition-based reconstruction
        expected = (np.transpose(b1.transformation_matrix()) @
                    fo.node_deform_list[0])
        assert_allclose(
            bd, expected,
            err_msg="bar_deform must equal T^T @ stacked nodal deformations.")
        # Check caching
        bd2 = fo.bar_deform[0]
        self.assertIs(
            bd, bd2,
            "bar_deform list should be cached on instance."
        )

    def test_system_deform_list(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        n2 = Node(4, 0, w='fixed')
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        fo = Solver(System([b1]))

        sys_def = fo.system_deform_list[0]
        node_def = fo.node_deform_list[0]
        assert_allclose(
            sys_def, node_def,
            err_msg="system_deform_list must equal node_deform_list when "
                    "transforming back to system coordinates."
        )

    def test_internal_forces(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1, n2 = Node(0, 0, u='fixed', w='fixed'), Node(3, 0, w='fixed')
        b1 = Bar(n1, n2, cross, mat,
                 point_loads=BarPointLoad(0, 10, 0, position=0.5))
        system = System([b1])
        system2 = System([Bar(n1, n2, cross, mat,
                              line_loads=BarLineLoad(1, 1))])
        # Check length equals number of elements
        self.assertEqual(
            len(Solver(system).internal_forces), len(system.mesh),
            'internal_forces must return one (6x1) array per bar.')
        # Check expected internal forces
        assert_allclose(
            Solver(system2).internal_forces,
            np.array([[[0], [-1.500000003], [0], [0], [-1.499999997], [0]]]),
            err_msg='Unexpected internal forces for the given uniform load.')

        fo1 = Solver(system2)
        elem = system2.mesh[0]
        k_loc = elem.stiffness_matrix(to_node_coord=False)
        f0_loc = elem.f0(to_node_coord=False) + elem.f0_point
        d_loc = fo1.bar_deform[0]
        assert_allclose(
            fo1.internal_forces[0], k_loc @ d_loc + f0_loc,
            err_msg="Internal forces must satisfy f' = k' * δ' + f0'."
        )

    def test_node_support_forces(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        # Include elastic supports to exercise the elastic term
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        n2 = Node(4, 0, u=200, phi=500,
                  w='fixed')
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        fo = Solver(System([b1]))

        k, d = fo.system_matrix, fo.node_deform
        f0, p0 = fo.f0, fo.p0
        elastic_vec = np.vstack(np.diag(fo.elastic_matrix))
        expected = k @ d + f0 - p0 - elastic_vec * d

        assert_allclose(
            fo.node_support_forces, expected,
            err_msg="node_support_forces must satisfy "
                    "Psupp = K*d + F0 - P0 - diag(elastic)*d."
        )

        self.assertEqual(
            fo.node_support_forces.shape, (fo.dof * len(fo.system.nodes()), 1),
            "node_support_forces must have size (dof * number_of_nodes, 1)."
        )

    def test_system_support_forces(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e-8, 0.1, 0.1,
                       0.1)
        # Two variants: rotation = 0 and rotation = 90deg (pi/2 rad)
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed', rotation=0)
        n2 = Node(4, 0, w='fixed')
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        fo0 = Solver(System([b1]))

        # With zero rotation, node/system reactions must coincide
        assert_allclose(
            fo0.system_support_forces, fo0.node_support_forces,
            err_msg="With rotation=0, system_support_forces "
                    "must equal node_support_forces."
        )

        # Now rotate node 1 by 90 degrees
        n1r = Node(0, 0, u='fixed', w='fixed', phi='fixed', rotation=np.pi / 2)
        b1r = Bar(n1r, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        fo = Solver(System([b1r]))

        local = fo.node_support_forces[:3, :]
        global_ = fo.system_support_forces[:3, :]

        self.assertFalse(
            np.allclose(local, global_),
            "With node rotation, transformed support forces "
            "must differ from local ones."
        )

    def test_hinge_modifier(self):
        # Check hinge_modifier returns correct relative displacements
        # for hinges
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1, n2 = (Node(0, 0, u='fixed', w='fixed', phi='fixed'),
                  Node(3, 0, w='fixed', u='fixed', phi='fixed'))
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1),
                 hinge_phi_i=True, hinge_w_i=True)
        # Case 1: hinge at i for w and phi
        assert_allclose(
            Solver(System([b1])).bar_deform_hinge,
            np.array([[[0], [2.485272459e-3], [1.104565538e-3],
                       [0], [0], [0]]]),
            err_msg='Unexpected hinge_modifier for hinge_phi_i and'
                    ' hinge_w_i at node i.')
        # Case 2: hinge at i for u and phi, and hinge at j for w
        b2 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1),
                 hinge_u_i=True, hinge_phi_i=True, hinge_w_j=True)
        assert_allclose(
            Solver(System([b2])).bar_deform_hinge,
            np.array([[[0], [0], [-2.209131075e-3],
                       [0], [4.142120766e-3], [0]]]),
            err_msg='Unexpected hinge_modifier for hinge_u_i & hinge_phi_i (i)'
                    ' and hinge_w_j (j).')

    def test_bar_deform_node_displacement(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1, n2 = (Node(0, 0, u='fixed', w='fixed', phi='fixed'),
                  Node(3, 0, displacements=NodeDisplacement(z=0.01, x=0.1)))
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        # Check no rotation: node displacements are not rotated
        assert_allclose(
            Solver(System([b1])).bar_deform_displacements,
            np.array([[[0], [0], [0], [0.1], [0.01], [0]]]),
            err_msg='If the bar inclination is 0, the node displacements are '
                    'not transformed.')
        # Check inclined bar: transformed local components
        n2 = Node(3, -10, displacements=NodeDisplacement(z=0.01, x=0.1))
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        assert_allclose(
            Solver(System([b1])).bar_deform_displacements,
            np.array([[[0], [0], [0],
                       [0.0191565257080], [0.0986561073760], [0]]]),
            err_msg='Local bar-end displacements must be the rotated version '
                    'of the nodal displacements for an inclined bar.')

    def test_bar_deform_list(self):
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        # Case with prescribed node displacement and hinges
        n1, n2 = (Node(0, 0, u='fixed', w='fixed', phi='fixed',
                       displacements=NodeDisplacement(1)),
                  Node(3, 0, w='fixed', u='fixed', phi='fixed'))
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1),
                 hinge_phi_i=True, hinge_w_i=True)
        assert_allclose(
            Solver(System([b1])).bar_deform_list,
            np.array([[[1], [2.485272459e-3], [1.104565538e-3],
                       [0], [0], [0]]]),
            err_msg='bar_deform_list must equal hinge_modifier + bar_deform + '
                    'bar_deform_displacements.')
        # Case with no hinges and fixed supports: zero deformation expected
        n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        assert_allclose(
            Solver(System([b1])).bar_deform_list,
            [np.zeros((6, 1))],
            err_msg='Total bar-end deformation must be zero with no hinges, '
                    'blocked supports, and no node displacements.')

    # def test_solvable(self):
    #     cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
    #     mat = Material(2.1e8, 0.1, 0.1, 0.1)
    #     # Negative case: no supports -> unsolvable
    #     n1, n2 = Node(0, 0), Node(3, 0)
    #     b1 = Bar(n1, n2, cross, mat,
    #              point_loads=BarPointLoad(0, 10, 0, position=0.5))
    #     self.assertFalse(
    #         Solver(System([b1])).solvable,
    #         'The system can not be solved if no supports are defined.')
    #     # Positive case: enough supports
    #     n1, n2 = Node(0, 0, u='fixed', w='fixed', phi='fixed'), Node(3, 0)
    #     fo = Solver(
    #         System([Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))]))
    #     self.assertTrue(
    #         fo.solvable,
    #         'The system must be solvable with sufficient supports.'
    #     )
    #     self.assertIsInstance(
    #         fo.calc, tuple,
    #         'calc must return a tuple when system is solvable.'
    #     )
    #     self.assertEqual(
    #         len(fo.calc), 2,
    #         'calc must return (bar_deform_list, internal_forces).'
    #     )


class TestSecondOrder(TestCase):

    def test_averaged_longitudinal_force(self):
        # Check averaged_longitudinal_force length/finite and order restoration
        cross = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        mat = Material(2.1e8, 0.1, 0.1, 0.1)
        n1, n2 = Node(0, 0, u='fixed', w='fixed', phi='fixed'), Node(3, 0)
        b1 = Bar(n1, n2, cross, mat, line_loads=BarLineLoad(1, 1))
        fo = SecondOrder(System([b1]))

        Lavg = fo.averaged_longitudinal_force
        self.assertEqual(
            len(Lavg), len(fo.system.mesh),
            'averaged_longitudinal_force must return one value per bar.'
        )
