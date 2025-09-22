
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from sstatics.core.preprocessing.system import System
from sstatics.core.postprocessing.results import BarResult, NodeResult


@dataclass
class SystemResult:
    r"""Represents the post-processing results of a structural system.

    This class aggregates discrete bar deformations, internal forces,
    nodal displacements, and support reactions from a solved
    :py:class:`System`. It provides convenient access to per-bar and
    per-node results for further analysis or visualization.

    Parameters
    ----------
    system : :py:class:`System`
        The analyzed structural system.
    bar_deform_list : list of :any:`numpy.ndarray`
        List of deformation vectors for each mesh bar in the system.
        Each deformation array corresponds to a mesh bar in
        :py:attr:`FirstOrder.bar_deform_list`.
    bar_internal_forces : list of :any:`numpy.ndarray`
        List of force vectors for each mesh bar in the system.
        Each force array corresponds to a mesh bar in
        :py:attr:`FirstOrder.bar_internal_forces`.
    node_deform : :any:`numpy.ndarray`
        A vector with dimensions (dof * number of nodes, 1) containing the
        resulting displacement of each node in its local coordinate system
        :py:attr:`FirstOrder.node_deform`.
    node_support_forces : :any:`numpy.ndarray`
        A vector with dimensions (dof * number of nodes, 1) containing the
        support reactions in the nodal coordinate system
        :py:attr:`FirstOrder.node_support_forces`.
    system_support_forces : :any:`numpy.ndarray`
        A vector with dimensions (dof * number of nodes, 1) that contains
        the support reactions referenced to the global coordinate system
        :py:attr:`FirstOrder.system_support_forces`.
    dof : int, default=3
        Number of degrees of freedom per node. Must be 3 for 2D systems
        (translations ux, uz and rotation φ). The value 6 would correspond
        to 3D systems (ux, uy, uz, φx, φy, φz), but 3D is currently not
        implemented.
    n_disc : :any:`int`, default=10
        Number of discrete evaluation points along each bar for discretisation
        of forces and deformations.

    Raises
    ------
    ValueError
        If the number of deformation or force vectors does not match the
        number of bars, if the node vectors have inconsistent shapes,
        or if `dof` or `n_disc` are invalid.
    TypeError
        If any of the input arrays are not instances of `numpy.ndarray`.


    Attributes
    ----------
    bars : list of :py:class:`BarResult`
        List containing the discrete results for each bar.
    nodes : list of :py:class:`NodeResult`
        List containing the displacement and support forces for each node.


    Examples
    --------
    >>> from sstatics.core.preprocessing import (
    >>>     Bar, BarLineLoad, CrossSection, Material, Node, System
    >>> )
    >>> from sstatics.core.solution import FirstOrder
    >>> from sstatics.core.postprocessing import SystemResult
    >>>
    >>> n1 = Node(0, 0, u='fixed', w='fixed')
    >>> n2 = Node(4, 0, w='fixed')
    >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
    >>> mat = Material(210000000, 0.1, 81000000, 0.1)
    >>> load = BarLineLoad(1, 1, 'z', 'bar', 'exact')
    >>> bar = Bar(n1, n2, cross, mat, line_loads=load)
    >>> system = System([bar])
    >>> solution = FirstOrder(system)
    >>> system_result = SystemResult(
    >>>                     system=system,
    >>>                     bar_deform_list=solution.bar_deform_list,
    >>>                     bar_internal_forces=solution.internal_forces,
    >>>                     node_deform=solution.node_deform,
    >>>                     node_support_forces=solution.node_support_forces,
    >>>                     system_support_forces=
    >>>                         solution.system_support_forces
    >>> )
    >>> for bar in system_result.bars:
    >>>     print(bar.x_coef)
    array([[ 0. -0.  0.]
           [ 0. -0. -0.]
           [ 0. -0.  0.]
           [ 0.  0.  0.]])
    """

    system: System
    bar_deform_list: list[np.ndarray]
    bar_internal_forces: list[np.ndarray]
    node_deform: np.ndarray
    node_support_forces: np.ndarray
    system_support_forces: np.ndarray
    dof: Literal[3, 6] = 3
    n_disc: int = 10

    def __post_init__(self):
        self._validation()
        self.bars = [
            BarResult(bar,
                      self.bar_deform_list[i],
                      self.bar_internal_forces[i],
                      self.n_disc)
            for i, bar in enumerate(self.system.mesh)
        ]
        self.nodes = [
            NodeResult(node,
                       self.node_deform[
                            i * self.dof: i * self.dof + self.dof],
                       self.node_support_forces[
                            i * self.dof: i * self.dof + self.dof],
                       self.system_support_forces[
                            i * self.dof: i * self.dof + self.dof])
            for i, node in enumerate(self.system.nodes())
        ]

    def _validation(self):
        # DOF check
        if self.dof not in (3, 6):
            raise ValueError(
                f'"dof" must be either 3 (2D space) or 6 (3D space),'
                f'got {self.dof}.'
            )
        if self.dof == 6:
            raise ValueError(
                '"dof=6" (3D systems) is not yet implemented.'
            )
        if self.n_disc < 1:
            raise ValueError('"n_disc" must be a positive integer.')

        # Check list lengths
        n_bars = len(self.system.mesh)
        if len(self.bar_deform_list) != n_bars:
            raise ValueError(
                f'Expected {n_bars} bar deformations, got '
                f'{len(self.bar_deform_list)}.'
            )
        if len(self.bar_internal_forces) != n_bars:
            raise ValueError(
                f'Expected {n_bars} bar force vectors, got '
                f'{len(self.bar_internal_forces)}.'
            )

        # Type and shape validation for lists
        for i, arr in enumerate(self.bar_deform_list):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f'bar_deform_list[{i}] must be a numpy.ndarray.'
                )
        for i, arr in enumerate(self.bar_internal_forces):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f'bar_internal_forces[{i}] must be a numpy.ndarray.'
                )

        # Shape checks for node-related arrays
        shapes = [self.node_deform.shape,
                  self.node_support_forces.shape,
                  self.system_support_forces.shape]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f'"node_deform", "node_support_forces", '
                f'and "system_support_forces" must all have the same shape, '
                f'got {shapes}.'
            )

        n_nodes = len(self.system.nodes(mesh_type="mesh"))
        expected_size = n_nodes * self.dof
        if self.node_deform.shape[0] != expected_size:
            raise ValueError(
                f'Expected {expected_size} rows for {n_nodes} nodes × '
                f'{self.dof} DOF, but got {self.node_deform.shape[0]}.'
            )

        if self.node_deform.ndim != 2 or self.node_deform.shape[1] != 1:
            raise ValueError(
                f'"node_deform" must be a column vector of shape '
                f'({expected_size}, 1), but got {self.node_deform.shape}.'
            )

    @cached_property
    def deforms_disc(self):
        r"""Discrete deformation vectors evaluated at discrete points.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the deformation vectors evaluated along
            the length of the corresponding mesh bar.

        See Also
        --------
        :py:attr:`BarResult.deform_disc`
        """
        return [result.deform_disc for result in self.bars]

    @cached_property
    def forces_disc(self):
        r"""Discrete force vectors evaluated at discrete points.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the force vectors evaluated along
            the length of the corresponding mesh bar.

        See Also
        --------
        :py:attr:`BarResult.forces_disc`
        """
        return [result.forces_disc for result in self.bars]
