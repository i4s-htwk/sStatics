
from unittest import TestCase

import numpy as np

from sstatics.core import NodeLoad


class TestNodeLoad(TestCase):

    def test_normalized_rotation(self):
        for rotation in (0, 1, 2, np.pi, 3, 4, 2 * np.pi):
            load = NodeLoad(1, 0, 1, rotation=rotation)
            with self.subTest(rotation=rotation):
                self.assertAlmostEqual(
                    rotation % (2 * np.pi), load.rotation,
                    msg="Load rotations should be normalized to the interval "
                    "[0, 2*Pi)."
                )

    # def test_no_rotation(self):
    #    for load in (NodeLoad(0, 0, 0), NodeLoad(1, 0, 1)):
    #        with self.subTest(node_load=load):
    #            self.assertEqual(
    #                load.rotate(0), load,
    #                "If there is no load and node rotation, the load rotation"
    #                "does not have any effect."
    #            )

    # def test_full_rotation(self):
    #    load = NodeLoad(1, 0, 1, rotation=4 * np.pi)
    #    for factor in range(2, 11, 2):
    #        with self.subTest(node_load=load, rotation=factor * np.pi):
    #            self.assertEqual(
    #                load.rotate(factor * np.pi), load,
    #                "Full rotations do not have an effect on the load."
    #            )

    # TODO: subtest with node rotation and no load rotation
    # TODO: subtest with both rotations
    # def test_rotation(self):
    #    load = NodeLoad(1, 0, 1)
    #    with self.subTest(node_load=load, rotation=np.pi / 2):
    #        self.assertTrue(np.isclose(
    #            load.rotate(np.pi / 2).vector,
    #            np.array([[np.cos(np.pi / 2)], [np.sin(np.pi / 2)], [1]])
    #        ).all())
