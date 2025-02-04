
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
        if self.coord == 'bar' and self.length == 'proj':
            raise ValueError(
                'If the used coordinate system is set to "bar", then the '
                'length cannot be set to "proj".'
            )

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
    """Creates a point load applied at a specific position along a bar.

    Parameters
    ----------
    x : :any:`float`
        The force component in the x-direction.
    z : :any:`float`
        The force component in the z-direction.
    phi : :any:`float`
        The moment applied along the beam.
    rotation : :any:`float`, default=0.0
        The rotation of the load components.
    position : :any:`float`, default=0.0
        Describes the relative position of the load along the bar. A value of
        `0` indicates the start of the bar, and `1` indicates the end of the
        bar.

    Notes
    -----
    This class models a point load applied to a bar (or beam) at a specific
    position. The load is applied in the x and z directions and includes a
    moment (phi) along the beam. The position is a normalized value between 0
    and 1, where 0 corresponds to the start of the bar and 1 corresponds to the
    end of the bar.

    Raises
    ------
    ValueError
        :py:attr:`position` has to be a value between 0 and 1.

    See Also
    --------
    :py:class:`NodePointLoad` and :py:class:`DegreesOfFreedom`
    """

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
