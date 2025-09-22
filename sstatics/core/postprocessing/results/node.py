
from dataclasses import dataclass

import numpy as np

from sstatics.core.preprocessing.node import Node


@dataclass
class NodeResult:
    r"""Represents the result vectors associated with a single node.

    This class stores the displacement and support reaction vectors
    for a given node, both in the nodal and global coordinate system.

    Parameters
    ----------
    node : :py:class:`Node`
        The node for which the results are stored.
    deform : numpy.ndarray
        Displacement vector of the node in its local coordinate system,
        with shape (3, 1).
    node_support : numpy.ndarray
        Support reaction vector in the nodal coordinate system,
        with shape (3, 1).
    system_support : numpy.ndarray
        Support reaction vector in the global coordinate system,
        with shape (3, 1).

    Raises
    ------
    TypeError
        If ``node`` is not an instance of :py:class:`Node` or any of the
        vectors are not ``numpy.ndarray``.
    ValueError
        If any of the vectors do not have shape (3, 1) or contain
        invalid values (NaN or infinite).
    TypeError
        If any of the vectors are not of type ``float32`` or ``float64``.

    Attributes
    ----------
    node : :py:class:`Node`
        Reference to the corresponding node object.
    deform : numpy.ndarray
        Local displacement vector of the node.
    node_support : numpy.ndarray
        Support reactions in the nodal coordinate system.
    system_support : numpy.ndarray
        Support reactions in the global coordinate system.

    Examples
    --------
    >>> from sstatics.core.preprocessing import Node
    >>> import numpy as np
    >>>
    >>> n1 = Node(0, 0, u="fixed", w="fixed", phi='fixed')
    >>> deform = np.zeros((3, 1))
    >>> node_support = np.array([[0.0], [10.0], [0.0]])
    >>> system_support = np.array([[0.0], [10.0], [0.0]])
    >>> node_result = NodeResult(
    ...     node=n1,
    ...     deform=deform,
    ...     node_support=node_support,
    ...     system_support=system_support
    ... )
    >>> node_result.deform
    array([[0.],
           [0.],
           [0.]])
    """

    node: Node
    deform: np.ndarray
    node_support: np.ndarray
    system_support: np.ndarray

    def __post_init__(self):
        self._validation()

    def _validation(self):
        if not isinstance(self.node, Node):
            raise TypeError('"node" must be an instance of Node.')

        for name, arr in [
            ("deform", self.deform),
            ("node_support", self.node_support),
            ("system_support", self.system_support),
        ]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f'"{name}" must be a numpy.ndarray.')
            if arr.shape != (3, 1):
                raise ValueError(
                    f'"{name}" must have shape (3, 1), got {arr.shape}.')
            if not np.isfinite(arr).all():
                raise ValueError(f'"{name}" contains NaN or infinite values.')
            if arr.dtype not in (np.float32, np.float64):
                raise TypeError(
                    f'"{name}" must be of type float32 or float64, '
                    f'got {arr.dtype}.')
