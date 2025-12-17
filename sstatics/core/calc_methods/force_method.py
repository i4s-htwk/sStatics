
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from typing import Literal

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.postprocessing.equation_of_work import EquationOfWork
from sstatics.core.calc_methods import FirstOrder
from sstatics.core.calc_methods import ReductionTheorem


@dataclass(eq=False)
class ForceMethod(ReductionTheorem):
    """Executes the Force Method

    Method to calculate internal forces, support forces and deformations of
    statically undetermined systems (n > 0). This method is based on the
    principle of virtual forces (PvF).

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the Force Method.
    """

    def __post_init__(self):
        """Initialize reduction workflow and state.

        Calls the base initializer and prepares an internal snapshot of
        the modifier after each release operation to track the reduced
        configuration.
        """
        self._system_modifier = SystemModifier(system=self.system)
        self._released_modifier = None
        self.logger.info("Force Method has successfully been created.")

    @cached_property
    def uls_systems(self):
        """Return virtual systems for the unit load states (ULS).

        First validates that the system is released and statically
        determinate (:meth:`_validate_system_ready`). All external
        loads are then cleared before generating the unit load
        systems, one for each redundant.

        Returns
        -------
        list[:any:`System`]
            List of generated systems, one per unit load case.
        """
        self.logger.info(
            "Accessing method to generate ULS (unit load state) systems.")
        self._validate_released_system()

        self._system_modifier.delete_loads()
        self.logger.debug(
            "All existing loads have been removed from the released system.")
        uls = self._system_modifier.create_uls_systems()
        self.logger.debug(
            f"{len(uls)} ULS systems have been successfully generated.")
        return uls

    @cached_property
    def solution_uls_systems(self):
        r"""Returns the first order analysis of the unit load states.

        A new :any:`FirstOrder` instance is created for each unit load system.
        Before solving, the system is meshed consistently with the real load
        state system, ensuring matching integration lengths for the internal
        work calculation.

        Returns
        -------
        list[:any:`FirstOrder`]
            List of generated FirstOrder instances, one per unit load case.
        """
        self.logger.info("Creating FirstOrder instances for the ULS systems.")
        solution = []
        for i, uls_system in enumerate(self.uls_systems):
            uls_system.create_mesh(
                self._system_modifier.division_positions_for(uls_system))
            self.logger.debug(
                f"ULS system {i} has been meshed for solution computation.")
            solution.append(FirstOrder(uls_system, debug=self.debug))
            self.logger.debug(
                f"FirstOrder instance for ULS system {i} created "
                f"successfully.")
        return solution

    @cached_property
    def solution_rls_system(self):
        r"""Returns the first order analysis of the released system in the
        real load state.

        A new :any:`FirstOrder` instance is created using the released
        system with real loads.

        Returns
        -------
        :any:`FirstOrder`
            FirstOrder instance operating on the released system with real
            loads.
        """
        self.logger.info(
            "Creating FirstOrder instances for the released system with real "
            "loads (real load state).")
        self.released_system.create_mesh(
            self._system_modifier.division_positions_mesh())
        self.logger.debug(
            "RLS system has been meshed for solution computation.")
        solution = FirstOrder(self.released_system, debug=self.debug)
        self.logger.debug(
            "FirstOrder instance for RLS system created successfully.")
        return solution

    @property
    def eow_vector(self):
        r"""Returns the vector of Equation of Work objects for the real load
        state.

        For each unit load system (ULS), an :any:`EquationOfWork` instance is
        created, representing the internal work between the released real load
        system (RLS) and the respective ULS. This vector forms the basis for
        computing the load coefficients used in the force method.

        Returns
        -------
        :any:`numpy.array`
            nx1 vector containing one EquationOfWork instance per unit
            load system. n = number of unit load states
        """
        self.logger.info(
            "Creating EquationOfWork objects for the real load state "
            "(EOW vector).")
        vector = np.array(
            [[EquationOfWork(self.solution_rls_system, uls, debug=self.debug)]
             for uls in self.solution_uls_systems
             ], dtype=object)
        self.logger.debug(
            f"EOW vector created with {len(vector)} EquationOfWork instances.")
        return vector

    @cached_property
    def eow_matrix(self):
        r"""Returns the matrix of Equation of Work objects for all ULS
        combinations.

        Creates a square matrix where each entry represents the internal work
        between a pair of unit load systems (ULS). This matrix forms the basis
        for evaluating the influence coefficients used in the force method.

        Returns
        -------
        :any:`numpy.array`
            Two-dimensional object array containing EquationOfWork instances,
            with one row and one column per redundant.
        """
        self.logger.info(
            "Creating the EquationOfWork matrix for all ULS combinations.")
        matrix = np.array(
            [[EquationOfWork(uls_i, uls_j, debug=self.debug)
              for uls_j in self.solution_uls_systems]
             for uls_i in self.solution_uls_systems],
            dtype=object
        )
        self.logger.debug(
            f"EOW matrix created with shape {matrix.shape}.")
        return matrix

    @cached_property
    def load_coef(self):
        r"""Calculate the load coefficients for the force method.

        Evaluates the value :math:`\delta_{i}` from each EquationOfWork object
        of the real load state with a unit load state. The resulting vector
        forms the right-hand side of the linear system in the force method.

        Returns
        -------
        :any:`numpy.array`
            Column vector (n×1) of floating-point load coefficients.
        """
        self.logger.info("Calculating load coefficients (δᵢ vector).")
        coef = np.array(
            [[eow[0].delta_ij] for eow in self.eow_vector],
            dtype=float
        )
        self.logger.debug(
            f"Load coefficient vector created with shape {coef.shape}.")
        return coef

    @cached_property
    def influence_coef(self):
        r"""Calculate the influence number matrix for the force method.

        Converts the EquationOfWork matrix into its numerical form by
        evaluating :math:`\delta_{ij}` for each entry. The resulting square
        matrix represents the coupling between redundants and forms the
        stiffness-like system matrix of the force method.

        Returns
        -------
        :any:`numpy.array`
            Two-dimensional square matrix of floating-point influence numbers.
        """
        self.logger.info("Calculating influence number matrix (δᵢⱼ matrix).")
        matrix = np.array(
            [[eow.delta_ij for eow in row]
             for row in self.eow_matrix],
            dtype=float
        )
        self.logger.debug(
            f"Influence coefficient matrix created with shape {matrix.shape}.")
        return matrix

    @property
    def redundants(self):
        r"""Solve the system of equations for the redundant forces.

        Solves the linear system

        .. math::  A \, x = -b

        where

        * :math:`A` is the influence number matrix from
          :meth:`influence_coef`,
        * :math:`b` is the load coefficient vector from :meth:`load_coef`.

        The resulting vector :math:`x` contains the magnitudes of all redundant
        forces.

        Returns
        -------
        :any:`numpy.array`
            One-dimensional array containing the solved redundant forces.
        """
        self.logger.info("Solving linear system for redundant forces.")
        result = np.linalg.solve(self.influence_coef, -self.load_coef)
        self.logger.debug(
            f"Redundant forces computed: {result.flatten().tolist()}")
        return result

    def _get_eow(self, i: int, j: int | None = None):
        """
        Access an EquationOfWork object from vector or matrix.

        If ``j`` is None, returns the i-th element of the EOW vector.
        Otherwise, returns the (i,j)-th element of the EOW matrix.

        Parameters
        ----------
        i : int
            Row index (ULS index) in the vector or matrix.
        j : int or None, optional
            Column index for the matrix case. Default is None (vector case).

        Returns
        -------
        EquationOfWork
            The requested EquationOfWork object.
        """
        self.logger.debug(f"Accessing EOW with i={i}, j={j}")
        if j is None:
            return self.eow_vector[i][0]
        return self.eow_matrix[i][j]

    def _get_work_matrix(self, kind: Literal['nodes', 'bars'],
                         i: int, j: int | None = None):
        """
        Retrieve the work matrix for nodes or bars.

        Parameters
        ----------
        kind : {'nodes', 'bars'}
            Type of structural object.
        i : int
            ULS row index in EOW vector or matrix.
        j : int or None, optional
            ULS column index for matrix case. Default is None (vector case).

        Returns
        -------
        ndarray
            The work matrix corresponding to the specified kind and indices.

        Raises
        ------
        ValueError
            If ``kind`` is not 'nodes' or 'bars'.
        """
        self.logger.debug(
            f"Retrieving work matrix of kind='{kind}' at i={i}, j={j}")
        eow = self._get_eow(i, j)

        if kind == 'bars':
            return eow.work_matrix_bars
        if kind == 'nodes':
            return eow.work_matrix_nodes

        msg = f"kind must be 'nodes' or 'bars', got {kind}"
        self.logger.error(msg)
        raise ValueError(msg)

    def _get_work_of(self, obj: Node | Bar, i: int, j: int | None = None,
                     sum: bool = True):
        """
        Retrieve the work contribution of a Node or Bar.

        Parameters
        ----------
        obj : Node or Bar
            The structural object whose work is requested.
        i : int
            ULS row index in EOW vector or matrix.
        j : int or None, optional
            ULS column index for matrix case. Default is None.
        sum : bool, optional
            If True (default), sum contributions for Bars with multiple mesh
            segments. Ignored for Nodes.

        Returns
        -------
        ndarray
            Work row (Node) or summed/staked work rows (Bar).

        Raises
        ------
        TypeError
            If ``obj`` is neither a Node nor a Bar.
        """
        self.logger.debug(
            f"Retrieving work for {obj} at i={i}, j={j}, sum={sum}")
        if isinstance(obj, Bar):
            matrix = self._get_work_matrix('bars', i, j)
            return self._work_of_bar(matrix=matrix, bar=obj, sum=sum)

        if isinstance(obj, Node):
            matrix = self._get_work_matrix('nodes', i, j)
            return self._work_of_node(matrix=matrix, node=obj)

        msg = f"Invalid object type: {type(obj).__name__}"
        self.logger.error(msg)
        raise TypeError(msg)

    def _validate_indices(self, i: int, j: int | None = None):
        """
        Validate ULS indices against available systems.

        Ensures that i and j are integers within the valid range of
        indices (0..n-1) where n is the number of unit load systems.

        Parameters
        ----------
        i : int
            Row index for vector or matrix.
        j : int or None, optional
            Column index for matrix case. Default is None.

        Raises
        ------
        TypeError
            If i or j are not integers.
        IndexError
            If i or j are out of the valid range.
        """
        max_idx = len(self.uls_systems) - 1
        self.logger.debug(
            f"Validating indices i={i}, j={j}, max_idx={max_idx}")

        if not isinstance(i, int):
            msg = f"i must be int, got {type(i).__name__}"
            self.logger.error(msg)
            raise TypeError(msg)
        if i < 0 or i > max_idx:
            msg = f"i={i} out of range 0..{max_idx}"
            self.logger.error(msg)
            raise IndexError(msg)

        if j is not None:
            if not isinstance(j, int):
                msg = f"j must be int or None, got {type(j).__name__}"
                self.logger.error(msg)
                raise TypeError(msg)
            if j < 0 or j > max_idx:
                msg = f"j={j} out of range 0..{max_idx}"
                self.logger.error(msg)
                raise IndexError(msg)

    def work_matrix(self,
                    kind: Literal['nodes', 'bars'],
                    uls_index_i: int = 0,
                    uls_index_j: int | None = None):
        """
        Public API to access the work matrix for nodes or bars.

        Validates indices and returns the corresponding work matrix.

        Parameters
        ----------
        kind : {'nodes', 'bars'}
            Type of work matrix to return.
        uls_index_i : int, optional
            Row index in EOW vector or matrix. Default is 0.
        uls_index_j : int or None, optional
            Column index for matrix case. Default is None.

        Returns
        -------
        ndarray
            Work matrix for nodes or bars at the specified indices.
        """
        self.logger.debug(f"Public access to work matrix kind='{kind}' "
                          f"uls_index_i={uls_index_i}, "
                          f"uls_index_j={uls_index_j}")
        self._validate_indices(uls_index_i, uls_index_j)
        return self._get_work_matrix(kind, uls_index_i, uls_index_j)

    def work_of(self,
                obj: Node | Bar,
                uls_index_i: int = 0,
                uls_index_j: int | None = None,
                sum: bool = True):
        """
        Public API to access the work of a Node or Bar.

        Validates indices and returns the work contribution of the specified
        object.

        Parameters
        ----------
        obj : Node or Bar
            Structural object whose work contribution is requested.
        uls_index_i : int, optional
            Row index in EOW vector or matrix. Default is 0.
        uls_index_j : int or None, optional
            Column index for matrix case. Default is None.
        sum : bool, optional
            If True (default), sum contributions for Bars with multiple
            mesh segments. Ignored for Nodes.

        Returns
        -------
        ndarray
            Work row (Node) or summed/staked rows (Bar).

        Raises
        ------
        TypeError
            If ``obj`` is neither a Node nor a Bar.
        """
        self.logger.debug(f"Public access to work of {obj} at "
                          f"uls_index_i={uls_index_i}, "
                          f"uls_index_j={uls_index_j}, "
                          f"sum={sum}")
        self._validate_indices(uls_index_i, uls_index_j)
        return self._get_work_of(obj, uls_index_i, uls_index_j, sum)

    def plot(
            self, system_mode: Literal['uls', 'rls'] = 'rls',
            uls_index: int | None = None,
            kind: Literal[
                'normal', 'shear', 'moment', 'u', 'w', 'phi',
                'bending_line'] = 'normal',
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            decimals: int = 2,
            sig_digits: int | None = None,
            n_disc: int = 10,
            mode: str = 'mpl',
            color: 'str' = 'red',
            show_load: bool = False,
            scale: int = 1
    ):
        r"""Plot internal forces or deformation results of either the system
        with real loads, the unit load states or the real load state.

        Parameters
        ----------
        systen_mode : {'uls', 'rls'}, default='rls'

            Defines whether the results of a unit load state or the real load
            state.
        uls_index : :any:`int` or None
            If the chosen mode is uls, then an index of the unit loads \
            system is needed to plot the chosen system.
        decimals : int, optional
            Number of decimals for label annotation.
        sig_digits: int | None, default=None
            Number of significant digits for label annotation.
        n_disc : int, default=10
            Number of subdivisions for result interpolation.
        mode : {'mpl', 'plotly'}, default='mpl'
            Chosen renderer
        color : str, default='red'
            Color of the plot
        show_load : bool, default=False
            Specifies whether the load is plotted.
        scale : int, default=1
            Scale factor for plot


        Raises
        ------
        ValueError
            If the system mode is invalid.
        """
        if system_mode not in ['uls', 'rls']:
            raise ValueError(
                f'Mode has to be either "uls" or "rls".'
                f'Got {mode} instead.'
            )
        if system_mode in ['rls'] and uls_index is not None:
            raise ValueError(
                f'If the mode is set to "rls", the '
                f'uls_index has to be set to None. Got mode: {mode} and'
                f'uls_index: {uls_index} instead.'
            )
        if system_mode == 'uls' and uls_index is None:
            raise ValueError(
                f'If the mode is set to "uls", the index can not be None.'
                f'Got mode: {mode} and index: {uls_index} instead.'
            )
        if system_mode == 'rls':
            return self.solution_rls_system.plot(
                kind, bar_mesh_type, decimals, sig_digits, n_disc, mode,
                color, show_load, scale)
        else:
            return self.solution_uls_systems[uls_index].plot(
                kind, bar_mesh_type, decimals, sig_digits, n_disc, mode,
                color, show_load, scale)

    def _validate_virtual_load(self, force):
        r"""Disallows virtual loads for this method.

        In the force method (KGV), virtual loads are not part of the
        computational procedure. Any attempt to define virtual loads is
        rejected.

        Raises
        ------
        ValueError
            Always raised, since virtual loads are incompatible with this
            method.
        """
        msg = "Virtual loads are not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def _validate_virtual_system(self):
        r"""Disallows virtual systems for this method.

        The force method does not make use of virtual systems. Any attempt to
        validate or access such a system is rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "There is not a virtual system being defined in this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def add_virtual_node_load(self, *args, **kwargs):
        r"""Disallows adding virtual node loads in the force method.

        Virtual node loads are a concept of the displacement method and are not
        used in the force method (KGV). Any attempt to add such loads is
        rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Virtual node loads are not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def add_virtual_bar_load(self, *args, **kwargs):
        r"""Disallows adding virtual bar loads in the force method.

        Since the force method does not employ virtual systems or virtual
        loads, defining bar loads in a virtual context is not permitted.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Virtual bar loads are not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def add_virtual_moment_couple(self, *args, **kwargs):
        r"""Disallows adding virtual moment couples in the force method.

        Virtual moment couples are incompatible with the solution strategy of
        the force method and are therefore rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Virtual moment couple is not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def virtual_system(self):
        r"""Disallows access to a virtual system.

        The force method does not define or use a virtual system. Accessing one
        is not permitted.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "There is not a virtual system in this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def solution_virtual_system(self):
        r"""Disallows access to a virtual solution.

        Virtual solutions are not generated in the force method. Any attempt to
        retrieve such a solution is rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "There is not a virtual solution in this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def deformation(self):
        r"""Disallows computation of single deformations.

        In the force method (KGV), deformations are not computed directly by
        querying individual values. Deformations are only implicitly evaluated
        through the Equation of Work. Accessing single deformations is
        therefore not supported.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Deformation cannot be computed for this method."
        self.logger.error(msg)
        raise ValueError(msg)
