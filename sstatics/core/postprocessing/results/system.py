
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core.preprocessing.system import System
from sstatics.core.postprocessing.results import BarResult


@dataclass
class SystemResult:
    r"""Calculates discrete result vectors for the bars of the provided system.

    Parameters
    ----------
    system : :py:class:`System`
        The statical system that was analyzed.
    deforms : list of :any:`numpy.ndarray`
        List of deformation vectors for each mesh bar in the system.
        Each deformation array corresponds to a mesh bar in
        :py:attr:`system.mesh`.
    forces : list of :any:`numpy.ndarray`
        List of force vectors for each mesh bar in the system.
        Each force array corresponds to a mesh bar in
        :py:attr:`system.mesh`.
    n_disc : :any:`int`, default=10
        Number of discrete evaluation points along each bar for discretisation
        of forces and deformations.

    Raises
    ------
    ValueError
        If the length of :py:attr:`deforms` or :py:attr:`forces` does not match
        the number of mesh in :py:attr:`system`.

    Attributes
    ----------
    bars : list of :py:class:`BarResult`
        List of each mesh bar with discrete results.

    Examples
    --------
    >>> from sstatics.core import (
    >>>     Bar, BarLineLoad, CrossSection, FirstOrder, Material, Node, System
    >>> )
    >>> from sstatics.core.postprocessing import SystemResult
    >>> n1 = Node(0, 0, u='fixed', w='fixed')
    >>> n2 = Node(4, 0, w='fixed')
    >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
    >>> mat = Material(210000000, 0.1, 81000000, 0.1)
    >>> load = BarLineLoad(1, 1, 'z', 'bar', 'exact')
    >>> bar = Bar(n1, n2, cross, mat, line_loads=load)
    >>> system = System([bar])
    >>> fo = FirstOrder(system)
    >>> results = fo.calc
    >>> system_result = SystemResult(system, results[0], results[1], n_disc=10)
    >>> for bar in system_result.bars:
    >>>     print(bar.x_coef)
    array([[ 0. -0.  0.]
           [ 0. -0. -0.]
           [ 0. -0.  0.]
           [ 0.  0.  0.]])
    """

    system: System
    deforms: list[np.ndarray]
    forces: list[np.ndarray]
    n_disc: int = 10

    def __post_init__(self):
        if len(self.system.mesh) != len(self.deforms):
            raise ValueError(
                'The number of bars in "system.mesh" does not match the '
                'number of entries in "deforms".'
            )
        if len(self.system.mesh) != len(self.forces):
            raise ValueError(
                'The number of bars in "system.mesh" does not match the '
                'number of entries in "forces".'
            )
        self.bars = [
            BarResult(bar, self.deforms[i], self.forces[i], self.n_disc)
            for i, bar in enumerate(self.system.mesh)
        ]

    @cached_property
    def length_disc(self):
        r"""Discrete evaluation points along the length of each mesh bar.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the discrete coordinate points along the
            corresponding bar where results are evaluated.

        See Also
        --------
        :py:attr:`length_disc` in :py:class:`BarResult`

        Examples
        --------
        >>> system_result = SystemResult(...)
        >>> system_result.length_disc
        """
        return [result.length_disc for result in self.bars]

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
        :py:attr:`deform_disc` in :py:class:`BarResult`
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
        :py:attr:`forces_disc` in :py:class:`BarResult`
        """
        return [result.forces_disc for result in self.bars]

    @cached_property
    def system_results_disc(self):
        r"""Tuple of deformation and force vectors evaluated at discrete
        points.

        Returns
        -------
        tuple of lists
            Tuple containing two lists:
                - List of deformation arrays evaluated at discrete points.
                - List of force arrays evaluated at discrete points.

        Examples
        --------
        >>> system_result = SystemResult(...)
        >>> deforms, forces = system_result.system_results_disc
        """
        return self.deforms_disc, self.forces_disc
