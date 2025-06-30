
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from sstatics.core import DegreesOfFreedom, transformation_matrix


@dataclass(eq=False)
class PointLoad(DegreesOfFreedom):
    r"""Create a point load applied to a statical system.

    This class is used to model a point load at a specific location, which can
    represent either a load applied at a node or along a bar. It includes
    force components in the x and z directions, as well as a rotational
    component (:math:`\phi`), which can represent a moment applied to either a
    node or a bar.

    Parameters
    ----------
    x : :any:`float`
        The force component in the x-direction. Represents the magnitude of the
        force along the x-axis.
    z : :any:`float`
        The force component in the z-direction. Represents the magnitude of the
        force along the z-axis.
    phi : :any:`float`
        The moment or rotational force applied at the load location. This can
        represent a torque at a node or a rotational component along a bar.
    rotation : :any:`float`, default=0.0
        The rotation of the load components, default is 0.0. This value
        specifies the initial angle (in rad) of the load's vector relative to a
        reference orientation.

    See Also
    --------
    :py:class:`NodePointLoad`
    :py:class:`BarPointLoad`

    Notes
    -----
        This class serves as the base class for modeling point loads.
    """

    rotation: float = 0.0

    def rotate(self, rotation: float):
        r"""Rotates the point load by a given angle and applies the rotation to
        the load vector.

        Parameters
        ----------
        rotation : :any:`float`
            The amount of rotation to apply, in radians. This is the angle by
            which the load vector is rotated relative to its current
            orientation.

        Returns
        -------
        :any:`numpy.array`
            The 3x1 transformed load vector after applying the rotation.

        Notes
        -----
            The rotation is performed by multiplying the transformation matrix
            with the load vector. The angle for the transformation matrix is
            calculated by subtracting the rotation of the point load from the
            input rotation:
            :python:`alpha = load.rotation - rotation`

            .. math ::
                \left(\begin{array}{ccc}
                \cos(\alpha) & \sin(\alpha) & 0 \\
                -\sin(\alpha) & \cos(\alpha) & 0 \\
                0 & 0 & 1
                \end{array}\right)
                \cdot
                \left(\begin{array}{c} x \\ z \\ \varphi \end{array}\right)
        """
        return transformation_matrix(self.rotation - rotation) @ self.vector


NodePointLoad = PointLoad
""" Alias of :py:class:`PointLoad` to make the use case of this class more
clear. """


@dataclass(eq=False)
class BarLineLoad:
    """Create a distributed line load applied to a structural bar element.

    Parameters
    ----------
    pi : :any:`float`
        The force at the start of the bar element.
    pj : :any:`float`
        The force at the end of the bar element.
    direction : {'x', 'z'}, default='z'
        The direction in which the load acts. Can be either:
            * :python:`'x'` : Load acts in the global/local x-direction.
            * :python:`'z'` : Load acts in the global/local z-direction.
    coord : {'bar', 'system'}, default='bar'
        Specifies the coordinate system in which the load is defined. \
        Can be either:
            * :python:`'bar'` : The load is applied in the local coordinate \
            system of the bar.
            * :python:`'system'` : The load is applied in the global \
            coordinate system.
    length : {'exact', 'proj'}, default='exact'
        Defines how the length of the load is considered. Can be either:
            * :python:`'exact'` : The exact length of the bar is used.
            * :python:`'proj'` : The projection of the bar length onto the \
            global coordinate system is used.

    Raises
    ------
    ValueError
        :py:attr:`direction` has to be either :python:`'x'` or \
        :python:`'z'`
    ValueError
        :py:attr:`coord` has to be either :python:`'bar'` or \
        :python:`'system'`
    ValueError
        :py:attr:`length` has to be either :python:`'exact'` or \
        :python:`'proj'`
    """

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
        """Computes the load vector for the bar element.

        Returns
        -------
        :any:`numpy.array`
            A 6x1 vector representing the distributed load at the bar element.

        Notes
        -----
        * The force at the start of the bar (`pi`) is assigned to index 0 \
        (x) or 1 (z).
        * The force at the end of the bar (`pj`) is assigned to index 3 \
        (x) or 4 (z).
        * If :py:attr:`pi` and :py:attr:`pj` are zero, then a 6x1 zero \
        vector is returned.
        """
        vec = np.zeros((6, 1))
        vec[0 if self.direction == 'x' else 1] = self.pi
        vec[3 if self.direction == 'x' else 4] = self.pj
        return vec

    def rotate(self, rotation: float):
        """Rotates the load vector if the coordinate system is global.

        Parameters
        ----------
        rotation : :any:`float`
            The rotation angle in rad.

        Returns
        -------
        :any:`numpy.array`
            The rotated load vector.

        Notes
        -----
        * If the load is defined in the local bar coordinate system,
          no rotation is applied.
        * If the load is defined in the global coordinate system, the
          transformation matrix accounts for rotation and projection
          effects.

        Examples
        --------
        >>> from sstatics.core import BarLineLoad
        >>> BarLineLoad(1, 1, 'z', 'bar', 'exact').rotate(0)
        array([[0], [1], [0], [0], [1], [0]])

        >>> from sstatics.core import BarLineLoad
        >>> import numpy as np
        >>> BarLineLoad(1, 1, 'z', 'system', 'proj').rotate(np.deg2rad(30))
        array([[-0.4330127], [0.75], [0], [-0.4330127], [0.75], [0]])
        """
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
    """Create a point load applied at a specific position along a bar.

    Parameters
    ----------
    x : :any:`float`
        The force component in the x-direction.
    z : :any:`float`
        The force component in the z-direction.
    phi : :any:`float`
        The moment applied along the beam.
    rotation : :any:`float`, default=0.0
        The rotation of the load components in rad.
    position : :any:`float`, default=0.0
        Describes the relative position of the load along the bar. A value of
        `0` indicates the start of the bar, and `1` indicates the end of the
        bar.

    Raises
    ------
    ValueError
        :py:attr:`position` has to be a value between 0 and 1.

    See Also
    --------
    :py:class:`PointLoad`
    :py:class:`NodePointLoad`

    Notes
    -----
        This class models a point load applied to a bar (or beam) at a specific
        position. The load is applied in the x and z directions and includes a
        moment (phi) along the beam. The position is a normalized value between
        0 and 1, where 0 corresponds to the start of the bar and 1 corresponds
        to the end of the bar.
    """

    position: float = 0.0

    def __post_init__(self):
        if not (0 <= self.position <= 1):
            raise ValueError("position must be between 0 and 1")

    # TODO: test
    def rotate(self):
        """Rotates the load vector based on its relative position.

        Returns
        -------
        :any:`numpy.array`
            A 6x1 array representing the rotated load vector.

        Notes
        -----
        * If :py:attr:`position` == 0, the load is placed at the start of\
         the bar.
        * If :py:attr:`position` == 1, the load is placed at the end of\
         the bar.
        * For intermediate positions (0 < :py:attr:`position` < 1), the\
         load contribution is considered elsewhere in the calculation and\
          does not appear directly in this vector. A 6x1 zero vector is \
          returned.

        Examples
        --------
        >>> from sstatics.core import BarPointLoad
        >>> load = BarPointLoad(x=5.0, z=0.0, phi=0.0, position=0.0)
        >>> load.rotate()
        array([[5.], [0.], [0.], [0.], [0.], [0.]])
        """
        vec = super().rotate(rotation=0.0)
        if self.position == 0:
            return np.vstack((vec, np.zeros((3, 1))))
        elif self.position == 1:
            return np.vstack((np.zeros((3, 1)), vec))
        else:
            return np.zeros((6, 1))
