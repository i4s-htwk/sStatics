
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from sstatics.core import DegreesOfFreedom, transformation_matrix


@dataclass(eq=False)
class PointLoad(DegreesOfFreedom):
    """ TODO """

    rotation: float = 0.0

    def rotate(self, rotation: float):
        """ TODO """
        return transformation_matrix(self.rotation - rotation) @ self.vector


NodePointLoad = PointLoad
""" Alias of :py:class:`PointLoad` to make the use case of this class more
clear. """


@dataclass(eq=False)
class BarLineLoad:
    """ TODO """

    pi: float
    pj: float
    direction: Literal['x', 'z'] = 'z'
    coord: Literal['bar', 'system'] = 'bar'
    length: Literal['exact', 'proj'] = 'exact'

    def __post_init__(self):
        if self.direction not in ('x', 'z'):
            raise ValueError('direction has to be either "x" or "z".')
        if self.coord not in ('bar', 'system'):
            raise ValueError('coord has to be either "bar" or "system".')
        if self.length not in ('exact', 'proj'):
            raise ValueError('length has to be either "exact" or "proj".')

    @cached_property
    def vector(self):
        """ TODO """
        vec = np.zeros((6, 1))
        vec[0 if self.direction == 'x' else 1] = self.pi
        vec[3 if self.direction == 'x' else 4] = self.pj
        return vec

    def rotate(self, rotation: float):
        """ TODO """
        if self.coord == 'bar':
            return self.vector
        sin, cos = np.sin(rotation), np.cos(rotation)
        m = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 0]])
        if self.length == 'proj':
            m = m @ np.diag([sin, cos, 0])
        m = np.vstack((
            np.hstack((m, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), m))
        ))
        return m @ self.vector


@dataclass(eq=False)
class BarPointLoad(PointLoad):
    """ TODO """

    # TODO: Documentation for variable position
    position: float = 0.0

    def __post_init__(self):
        if not (0 <= self.position <= 1):
            raise ValueError("position must be between 0 and 1")

    # TODO: test, docu
    def rotate(self):
        """ TODO """
        vec = super().rotate(rotation=0.0)
        if self.position == 0:
            return np.vstack((vec, np.zeros((3, 1))))
        elif self.position == 1:
            return np.vstack((np.zeros((3, 1)), vec))
        else:
            return np.zeros((6, 1))
