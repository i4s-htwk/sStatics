
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from sstatics.core import NodeDisplacement, NodePointLoad


@dataclass(eq=False)
class Node:
    """Create a node for a statical system.

    Parameters
    ----------
    x, z : :any:`float`
        Coordinates of the node in the x and z directions.
    rotation : :any:`float`, default=0.0
        Initial rotation of the node.
    u, w, phi : {'free', 'fixed'} or :any:`float`, default='free'
        Boundary conditions (supports) for the node:
            * :python:`'free'`: no constraint
            * :python:`'fixed'`: fully constrained
            * :any:`float`: represents a spring support with a specific \
            stiffness value (nib width).
    displacements : :any:`tuple`, default=()
        Prescribed displacements acting on the node. hese should be instances
        of the :py:class:`NodeDisplacement`.
    loads : :any:`tuple`, default=()
        Point loads acting on the node. These should be instances of the
        :py:class:`NodePointLoad`.

    Raises
    ------
    ValueError
        :py:attr:`u`, :py:attr:`w` and :py:attr:`phi` have to be either
        :python:`'free'`, :python:`'fixed'` or a real number.
    ValueError
        :py:attr:`u`, :py:attr:`w` or :py:attr:`phi` are set to zero. A spring
        with a nib width of zero behaves like a free fixture and therefore the
        value need to be set to :python:`'free'`.
    """

    x: float
    z: float
    rotation: float = 0.0
    u: Literal['free', 'fixed'] | float = 'free'
    w: Literal['free', 'fixed'] | float = 'free'
    phi: Literal['free', 'fixed'] | float = 'free'
    displacements: (
        tuple[NodeDisplacement, ...] | list[NodeDisplacement] |
        NodeDisplacement
    ) = ()
    loads: (
        tuple[NodePointLoad, ...] | list[NodePointLoad] | NodePointLoad
    ) = ()

    def __post_init__(self):
        for param in (self.u, self.w, self.phi):
            if isinstance(param, str) and param not in ('fixed', 'free'):
                raise ValueError(
                    f'"{param}" is an invalid argument. Has to be either '
                    f'"fixed" or "free" or a real number.'
                )
            elif param == 0:
                raise ValueError(
                    'Please set u, w or phi to "free" instead of zero.'
                )
        if isinstance(self.displacements, NodeDisplacement):
            self.displacements = self.displacements,
        self.displacements = tuple(self.displacements)
        if isinstance(self.loads, NodePointLoad):
            self.loads = self.loads,
        self.loads = tuple(self.loads)

    @cached_property
    def displacement(self):
        """The overall node displacement as a 3x1 vector.

        Returns
        -------
        :any:`numpy.array`
            Sum of all displacements specified in :py:attr:`displacements`.

        See Also
        --------
        :py:class:`NodeDisplacement`

        Notes
        -----
            If no displacements were specified, then a 3x1 zero vector is
            returned.

        Examples
        --------
        >>> from sstatics.core import Node
        >>> Node(1, 2).displacement
        array([[0], [0], [0]])

        >>> from sstatics.core import NodeDisplacement
        >>> displacements = (NodeDisplacement(1.5, 2, 0.5),
        >>>                  NodeDisplacement(-2, 3, -0.3))
        >>> Node(-1, 3, displacements=displacements).displacement
        array([[-0.5], [5], [0.2]])
        """
        if len(self.displacements) == 0:
            return np.array([[0], [0], [0]])
        return np.sum([d.vector for d in self.displacements], axis=0)

    @cached_property
    def load(self):
        r"""Rotate the node loads.

        Every load from :py:attr:`loads` is rotated by the node's rotation by
        calling :any:`PointLoad.rotate`. So the resulting rotation angle is
        calculated by :python:`alpha = load.rotation - node.rotation`. Thus
        each rotated load is computed by

        .. math::
            \left(\begin{array}{c}
            \cos(\alpha) & \sin(\alpha) & 0 \\
            -\sin(\alpha) & \cos(\alpha) & 0 \\
            0 & 0 & 1
            \end{array}\right) \cdot
            \left(\begin{array}{c} x \\ z \\ \varphi \end{array}\right)

        and summed up afterward.

        Returns
        -------
        :any:`numpy.array`
            Sum of all loads specified in :py:attr:`loads`.

        See Also
        --------
        :py:class:`NodePointLoad`

        Notes
        -----
            If no loads were specified, then a 3x1 zero vector is returned.

        Examples
        --------
        >>> from sstatics.core import Node, NodePointLoad
            >>> import numpy
        >>> load = NodePointLoad(1, 2, 0.5, rotation=2 * numpy.pi)
        >>> Node(6, 5, rotation=numpy.pi, loads=(load,)).load
        array([[-1], [-2], [0.5]])
        """
        if len(self.loads) == 0:
            return np.array([[0], [0], [0]])
        return np.sum(
            [load.rotate(self.rotation) for load in self.loads], axis=0
        )

    @cached_property
    def elastic_support(self):
        r"""Sets the stiffness values from the support conditions on the
        diagonal of the matrix.

        To account for the elastic support of nodes, the support conditions
        of :py:attr:`u`, :py:attr:`w`, and :py:attr:`phi` are considered and
        placed on the diagonal of a zero matrix.

        Returns
        -------
        :any:`numpy.array`
            A 3x3 zero matrix with the diagonal populated by the values of
            :py:attr:`u`, :py:attr:`w` and :py:attr:`phi`.

        Notes
        -----
            If an attribute is a string, its value is set to zero.

            .. math::
                \begin{bmatrix}
                       u & 0 & 0 \\
                       0 & w & 0 \\
                       0 & 0 & \varphi \\
                   \end{bmatrix}

        Examples
        --------
        >>> from sstatics.core import Node
        >>> Node(0, 0, u='fixed', w='fixed', phi='free').elastic_support
        array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        >>> Node(0, 0, u=100, w=200, phi=1000).elastic_support
        array([[100, 0, 0], [0, 200, 0], [0, 0, 1000]])
        """
        u = 0 if isinstance(self.u, str) else self.u
        w = 0 if isinstance(self.w, str) else self.w
        phi = 0 if isinstance(self.phi, str) else self.phi
        return np.diag([u, w, phi])

    def same_location(self, other):
        """Determine if two nodes have exactly the same :py:attr:`x`- and
        :py:attr:`z`-coordinates.

        Parameters
        ----------
        other : :any:`Node`
            Second node to compare coordinates to.

        Returns
        -------
        :any:`bool`
            :python:`True` if the nodes have exactly the same coordinates,
            :python:`False` otherwise.

        Examples
        --------
        >>> from sstatics.core import Node
        >>> node = Node(1, 2)
        >>> node.same_location(Node(1, 2))
        True
        >>> node.same_location(Node(1, -2))
        False
        """
        return self.x == other.x and self.z == other.z
