
from dataclasses import dataclass, replace, field, fields
from functools import cached_property
import numpy as np
from typing import List, Literal, Optional

from sstatics.core.postprocessing.results import DifferentialEquationSecond

from sstatics.core.logger_mixin import LoggerMixin
from sstatics.core.utils import get_differential_equation

from sstatics.core.preprocessing import Bar, BarSecond, System, SystemModifier
from sstatics.core.solution.solver import Solver


@dataclass
class SecondOrder(LoggerMixin):
    """Executes second-order structural analysis for the given system.

    This class performs analysis according to second-order theory,
    considering geometric nonlinearities that arise from large
    displacements or axial forces. Two solution strategies are available:
    a matrix-based approach and an iterative approach.

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed according to second-order theory.
    debug : bool, default=False
        Enable debug logging for intermediate steps.

    Notes
    -----
    **1. Matrix-based approach**

    This formulation replaces each first-order bar element by a modified
    second-order bar element (:class:`BarSecond`). Depending on the selected
    approach, the geometric stiffness contributions are incorporated via:

    - ``"analytic"``: analytical second-order stiffness matrix
    - ``"taylor"``: Taylor series expansion
    - ``"p_delta"``: P–Δ geometric stiffness formulation

    After selecting the formulation through :meth:`matrix_approach`, an
    internally modified :class:`System` instance is created. This modified
    system is then passed into a standard :class:`Solver`, which performs
    load–displacement computation under linear assumptions but with modified
    stiffness.

    **2. Iterative approach**

    The iterative method computes geometric nonlinearity by repeatedly:

    1. Solving the current system for nodal displacements
    2. Updating the geometry
    3. Recalculating the updated system
    4. Checking convergence via displacement change (``tolerance``)

    Depending on ``result_type``:

    - ``"cumulative"`` — results represent total deformation/state at iteration
        *i*
    - ``"incremental"`` — results store the difference between iteration *i*
      and iteration *i − 1*

    All iteration steps, including geometric updates are
    stored in ``_iteration_results``.

    References
    ----------
    .. [1] R. Dallmann. "Baustatik 3: Theorie II. Ordnung und
            computerorientierte Methoden der Stabtragwerke". 2. Auflage, 2015.

    .. [2] H. Werkle. "Finite Elemente in der Baustatik: Statik und Dynamik der
            Stab- und Flächentragwerke". 3. Auflage, 2008.

    .. [3] W.B. Krätzig, R. Harte, C. Könke, Y.S. Petryna. "Tragwerke 2:
            Theorie und Berechnungsmethoden statisch unbestimmter
            Stabtragwerke". 5. Auflage, 2019.

    .. [4] C. Spura. "Einführung in die Balkentheorie nach Timoshenko und
            Euler- Bernoulli". 2019.

    .. [5] H. Rothert, V. Gensichen. "Nichtlineare Stabstatik: Baustatische
            Grundlagen und Anwendungen". 1. Auflage, 1987.
    """
    system: System
    debug: bool = False

    _modified_system_matrix: Optional[System] = field(init=False, default=None)
    _solution_matrix: Optional[Solver] = field(init=False, default=None)
    _iteration_results: List[dict] = field(
        default_factory=list, init=False, repr=False)
    _iteration_mode: Literal['cumulative', 'incremental'] | None = field(
        init=False, default=None)

    def __post_init__(self):
        self.logger.info("SecondOrder successfully created.")

    def matrix_approach(
            self, approach: Literal['analytic', 'p_delta', 'taylor']):
        r"""Configure and initialize the matrix-based second-order analysis.

        This method selects the second-order formulation to be applied to each
        structural element. A modified :class:`System` instance is constructed
        internally in which each bar is replaced by a :class:`BarSecond`
        element using the chosen theory.

        Parameters
        ----------
        approach : {'analytic', 'p_delta', 'taylor'}
            Selects the second-order formulation for the modified stiffness:

            - ``'analytic'``
              Closed-form exact geometric stiffness
              effects.
            - ``'p_delta'``
              P–Δ geometric stiffness formulation.
            - ``'taylor'``
              Taylor-series expansion

        Notes
        -----
        After calling this method, the modified system can be accessed via
        :attr:`system_matrix_approach` and the corresponding solver via
        :attr:`solver_matrix_approach`.

        Raises
        ------
        ValueError
            If `approach` is not one of the supported options.
        """
        self.logger.info(f"Matrix approach selected: '{approach}'.")

        if approach not in ['analytic', 'p_delta', 'taylor']:
            msg = (f'Matrix approach has to be either "analytic", "taylor" or '
                   f'"p_delta". Got "{approach}" instead.')
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Resetting previous matrix-based results.")
        self._solution_matrix = None
        self._modified_system_matrix = None

        self.logger.debug(
            "Converting bars to second-order elements (BarSecond).")
        self.logger.debug(f"Number of bars: {len(self.system.bars)}")

        self._modified_system_matrix = System(self._convert_bars(approach))
        self.logger.info("Modified second-order system successfully created.")

    @property
    def solver_matrix_approach(self):
        r"""Return the solver associated with the chosen matrix-based
        second-order formulation.

        A new :class:`Solver` instance is created on first access, using the
        internally stored modified second-order system generated via
        :meth:`matrix_approach`. Internal bar forces of the solver are replaced
        by second-order transformed forces.

        Returns
        -------
        :class:`Solver`
            Solver instance operating on the modified second-order system.

        Raises
        ------
        AttributeError
            If no matrix-based system is available. Users must call
            :meth:`matrix_approach` before accessing this property.
        """
        self.logger.info("Accessing solver for matrix-based analysis.")
        if self._modified_system_matrix is None:
            msg = ("The modified system has not been created yet. "
                   "Call `matrix_approach(...)` before accessing "
                   "`solver_matrix_approach`.")
            self.logger.error(msg)
            raise AttributeError(msg)

        if self._solution_matrix is None:
            solver = Solver(self._modified_system_matrix, debug=self.debug)
            self.logger.info("Creating new solver for modified matrix system.")

            object.__setattr__(solver, 'internal_forces',
                               self._transform_internal_forces(solver))
            self.logger.debug(
                "The 'internal_forces' attribute of the Solver has been "
                "updated with the transformed second-order forces.")

            self._solution_matrix = solver
        self.logger.debug("Returning cached matrix-based solver instance.")

        return self._solution_matrix

    @property
    def system_matrix_approach(self):
        r"""Return the constructed second-order system.

        This system represents the structure where each bar element is replaced
        by its second-order equivalent using the formulation selected via
        :meth:`matrix_approach`.

        Returns
        -------
        :class:`System`
            The modified second-order system.

        Raises
        ------
        AttributeError
            If no matrix-based system has been computed yet.
        """
        self.logger.info("Accessing modified system of matrix-based analysis.")

        if self._modified_system_matrix is None:
            msg = ("The modified system has not been created yet. "
                   "Call `matrix_approach()` before accessing "
                   "`system_matrix_approach`.")
            self.logger.error(msg)
            raise AttributeError(msg)

        return self._modified_system_matrix

    def iterative_approach(
            self, iterations: int = 10, tolerance: float = 1e-3,
            result_type: Literal['cumulative', 'incremental'] = 'cumulative'):
        r"""Execute an iterative geometric–nonlinear second-order analysis.

        This method performs repeated updates of the system geometry based on
        newly computed nodal displacements. The process continues until
        convergence is reached or the maximum number of iterations is
        exhausted.

        Parameters
        ----------
        iterations : int, default=10
            Maximum number of allowed iteration steps.
        tolerance : float, default=1e-3
            Convergence threshold based on displacement changes between
            successive iterations.
        result_type : {'cumulative', 'incremental'}, default='cumulative'
            Determines the stored format of iteration results:

            - ``'cumulative'``: total values at each iteration
            - ``'incremental'``: differences between iteration *i* and *i–1*

        Notes
        -----
        The results of each iteration are stored in ``_iteration_results`` and
        can be accessed via :meth:`solver_iteration_cumulative()` for the
        'cumulative' result type or via :meth:`iterative_growth()` for the
        'incremental' result type.

        Raises
        ------
        ValueError
            If `iterations` is not a positive integer.
        ValueError
            If `result_type` is invalid.
        """
        self.logger.info(
            f"Starting iterative second-order analysis "
            f"(iterations={iterations}, tolerance={tolerance}, "
            f"mode={result_type})."
        )

        if not isinstance(iterations, int):
            msg_1 = (f'The number of iterations must be of type int. '
                     f'The current type is "{type(iterations)}".')
            self.logger.error(msg_1)
            raise ValueError(msg_1)

        if iterations <= 0:
            msg_2 = 'The number of iterations has to be greater than zero.'
            self.logger.error(msg_2)
            raise ValueError(msg_2)

        if result_type not in ('cumulative', 'incremental'):
            msg_3 = (f'Iteration mode has to be either "cumulative" or '
                     f'"incremental". Got "{result_type}" instead.')
            self.logger.error(msg_3)
            raise ValueError(msg_3)

        self.logger.debug("Resetting previous iteration results.")
        self._iteration_results = []
        self._iteration_mode = None

        self.logger.info("Running geometric-nonlinear iteration loop...")
        self._iteration_results = self._run_iteration(
            iterations, tolerance, result_type)
        self._iteration_mode = result_type
        self.logger.debug("Completed iterative approach.")

    def solver_iteration_cumulative(
            self, iteration: int = -1):
        r"""Return a solver corresponding to a specific iteration step for
        the iteration mode 'cumulative'.

        The solver is reconstructed by using the system state stored in the
        iteration results. Negative indices may be used to access iterations
        from the end (``-1`` = last iteration).

        Parameters
        ----------
        iteration : int, default=-1
            Index of the iteration to extract. Supports negative indexing.

        Returns
        -------
        :class:`Solver`
            Solver configured for the geometry and internal forces at the
            selected iteration.

        Raises
        ------
        AttributeError
            If no iteration has been performed yet.
        ValueError
            If the requested iteration index does not exist.
        """
        self.logger.info(
            "Calling the solver for the iterative approach of iteration: "
            f"{iteration}")

        if not self._iteration_results:
            msg = ("No iterations have been performed yet. "
                   "Run iterative_approach() first.")
            self.logger.error(msg)
            raise AttributeError(msg)
        try:
            entry = self._iteration_results[iteration]
        except Exception as exc:
            self.logger.error(
                f"Failed to find Iteration {exc}.",
                exc_info=True
            )
            raise
        self.logger.debug(
            "Solver created for requested iteration.")
        return Solver(entry["system"], debug=self.debug)

    def results_iterative_growth(
            self, iteration: int = -1, difference: Literal[
                'internal_forces', 'bar_deform_list', 'node_deform',
                'node_support_forces', 'system_support_forces']
            = 'internal_forces'
    ):
        r"""Retrieve the incremental growth result for a specific iteration
        step.

        This method returns a specific difference field for the given
        iteration, if the chosen iteration mode is incremental.

        The function reconstructs the incremental data from the stored
        iteration results. Negative indices are supported
        (e.g., ``-1`` refers to the last iteration).

        Parameters
        ----------
        iteration : int, default=-1
            The iteration index to extract. Supports negative indexing to count
            from the end of the iteration history.

        difference : {'internal_forces', 'bar_deform_list', 'node_deform',
                      'node_support_forces', 'system_support_forces'},
                      default='internal_forces'
            Specifies which incremental data field to return when using the
            incremental iteration mode.

        Returns
        -------
        Any
            The incremental difference dataset associated with the chosen
            `difference` type.

        Raises
        ------
        AttributeError
            If no iterations have been performed yet.

        ValueError
            - If the function is called while the iteration mode is
            ``cumulative``.
            - If the requested `difference` type is invalid.

        IndexError
            If the requested iteration index does not exist.
        """
        self.logger.info(
            "Calling the iterative growth for the incremental iterative "
            f"approach of iteration: {iteration} and the chosen iterative "
            f"growth of: {difference}")

        if not self._iteration_results:
            msg = ("No iterations have been performed yet. "
                   "Run iterative_approach() first.")
            self.logger.error(msg)
            raise AttributeError(msg)

        if self._iteration_mode == 'cumulative':
            msg = ("If the chosen result_type is 'cumulative', it is not "
                   "possible to show the iterative growth.")
            self.logger.error(msg)
            raise ValueError(msg)

        if difference not in [
            'internal_forces', 'bar_deform_list', 'node_deform',
                'node_support_forces', 'system_support_forces']:
            msg = ("difference has to be either 'internal_forces', "
                   "'bar_deform_list', 'node_deform', 'node_support_forces' "
                   "or 'system_support_forces'")
            self.logger.error(msg)
            raise ValueError(msg)
        try:
            entry = self._iteration_results[iteration]
        except Exception as exc:
            self.logger.error(
                f"Failed to find Iteration {exc}.",
                exc_info=True
            )
            raise
        self.logger.debug(
            f"The iterative growth of {difference} is:\n "
            f"{entry[difference + '_diff']}.")

        return entry[difference + '_diff']

    @property
    def max_shift(self):
        """Return the maximum nodal shift magnitude for each iteration.

        Retrieves the maximum nodal shift values computed during each step
        of the iterative procedure. These values indicate how much the nodal
        geometry changed from one iteration to the next and are typically used
        to assess convergence behavior.

        Returns
        -------
        list of `float`
            List containing the maximum nodal shift magnitude for each
            iteration. The index of each entry corresponds to the respective
            iteration step.

        Raises
        ------
        AttributeError
            If no iteration results are available. This occurs when the
            iterative procedure has not been executed yet.
        """
        self.logger.info("Calling the max shift.")
        if not self._iteration_results:
            msg = ("No iterations have been performed yet. "
                   "Run iterative_approach() first.")
            self.logger.error(msg)
            raise AttributeError(msg)

        max_shift = []
        for i in range(self.iteration_count):
            max_shift.append(self._iteration_results[i]['max_shift'])
            self.logger.info(
                f"[Iteration {i}] max shift: {max_shift[i]}.")
        return max_shift

    def system_iterative(self, iteration: int = -1):
        r"""Return the structural system for a specific iteration step.

        Negative indices follow Python semantics (``-1`` returns the last
        iteration). The returned :class:`System` reflects updated bar lengths,
        orientations and nodal positions at that iteration.

        Parameters
        ----------
        iteration : int, default=-1
            Iteration index. Negative values count from the end.

        Returns
        -------
        :class:`System`
            The system geometry of the selected iteration step.

        Raises
        ------
        RuntimeError
            If no iterations have been carried out.
        ValueError
            If the index is out of range.
        """
        self.logger.info(f"Returning system for iteration index {iteration}")

        if not self._iteration_results:
            msg = ("No iterations have been performed yet. "
                   "Run get_iteration_results() first.")
            self.logger.error(msg)
            raise RuntimeError(msg)
        try:
            return self._iteration_results[iteration]["system"]
        except Exception as exc:
            self.logger.error(
                f"Failed to find Iteration {exc}.",
                exc_info=True
            )
            raise

    @property
    def iteration_count(self):
        r"""Return the total number of performed iteration steps.

        Returns
        -------
        `int`
            Number of computed iterations.

        Raises
        ------
        RuntimeError
            If no iterative analysis has been performed yet.
        """
        if not self._iteration_results:
            msg = ("No iterations have been performed yet. "
                   "Run get_iteration_results() first.")
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.debug(f"Total iterations available: "
                          f"{len(self._iteration_results)}")
        return len(self._iteration_results)

    @cached_property
    def averaged_longitudinal_force(self):
        r"""Transformation of normal and shear forces to longitudinal force.

        The calculation of the longitudinal force is necessary to calculate
        systems according to the matrix approach for second-order theory.

        Since equilibrium according to second-order theory is determined on the
        deformed system, it is common practice to replace the normal and shear
        forces with their statically equivalent transverse force :math:`T` and
        longitudinal force :math:`L`. The average longitudinal force is
        required in second-order theory to adjust the element stiffness matrix
        and the load vectors.

        Notes
        -----
            The transformation is performed using the following equations:

            At the start of the bar:

            .. math:: L_{i} = N_{i} \cdot \cos(\varphi_{i}) + V_{i} \cdot \
                            \sin(\varphi_{i})

            At the end of the bar:

            .. math:: L_{j} = N_{j} \cdot \cos(\varphi_{j}) + V_{j} \cdot \
                    \sin(\varphi_{j})

            Subsequently, the average longitudinal force over the entire bar is
            calculated as:

            .. math:: L_{avg} = \dfrac{L_{i} + L_{j}}{2}

            This simplifies the assumption that the longitudinal force is
            constant throughout the length of the bar.

            If the longitudinal force is not constant, discretization errors
            may occur. In such cases, it is recommended to divide the bar into
            multiple segments, which improves the accuracy of the calculation.
        """
        self.logger.info(
            "Computing averaged longitudinal forces for all bars.")
        solution = Solver(self.system, debug=self.debug)
        self.logger.debug(
            "Created solver for original system to obtain bar "
            "forces and deformations.")
        l_avg = []
        for i, (deform, force) in enumerate(
                zip(solution.bar_deform_list,
                    solution.internal_forces)):
            self.logger.info(
                f"[Bar {i}] Calculating averaged longitudinal force.")
            l_i = -force[0, 0] * np.cos(deform[2, 0]) - force[1, 0] * np.sin(
                deform[2, 0])
            l_j = force[3, 0] * np.cos(deform[5, 0]) + force[4, 0] * np.sin(
                deform[5, 0])
            self.logger.debug(f"[Bar {i}] L_i={l_i}, L_j={l_j}")
            l_av = (l_i + l_j) / 2
            self.logger.info(f"[Bar {i}] L_avg={l_av}")
            l_avg.append(l_av)
        return l_avg

    def _convert_bars(
            self,
            approach: Literal['analytic', 'taylor', 'p_delta'] = 'analytic'):
        """Convert the system's bars into second-order bars for the
        matrix-based approach.

        Each bar is transformed into a :class:`BarSecond` instance
        corresponding to the selected second-order theory formulation. The
        conversion requires the longitudinal (axial) force of each bar,
        provided via :py:attr:`averaged_longitudinal_force`.

        Parameters
        ----------
        approach : {'analytic', 'taylor', 'p_delta'}, default='analytic'
            Defines which second-order formulation to use for the conversion.

        Returns
        -------
        list of :class:`BarSecond`
            List of converted bar objects ready for second-order matrix
            analysis.
        """
        self.logger.info(
            f"Converting bars to second-order elements "
            f"using approach='{approach}'.")
        bars = list(self.system.bars)
        bar_field_names = {f.name for f in fields(Bar)}

        for i, f_axial in enumerate(self.averaged_longitudinal_force):
            self.logger.debug(
                f"[Bar {i}] Using averaged axial force f_axial={f_axial}")

        converted = [
            BarSecond(
                **{k: getattr(bar, k) for k in bar_field_names},
                approach=approach,
                f_axial=f_axial
            )
            for bar, f_axial in zip(bars, self.averaged_longitudinal_force)
        ]

        self.logger.info("Bar conversion completed.")
        return converted

    def _transform_internal_forces(self, solver):
        """Rotates end forces using current bar-end rotations (phi_i, phi_j).

        The local force components are mapped with the instantaneous bar-end
        rotations to ensure second-order consistent reporting.

        Returns
        -------
        list of `np.ndarray`
            List of (6×1) force vectors per bar in local bar coordinates.
        """
        self.logger.info(
            "Transforming internal forces based on current bar-end rotations.")
        forces_list = []
        for i, (deform, force) in (
                enumerate(zip(solver.bar_deform_list,
                              solver.internal_forces))):
            self.logger.info(f"[Bar {i}] Transforming internal forces.")
            phi_i, phi_j = deform[2, 0], deform[5, 0]
            l_i, l_j = -force[0, 0], force[3, 0]
            t_i, t_j = -force[1, 0], force[4, 0]
            self.logger.debug(f"[Bar {i}] φ_i={phi_i}, φ_j={phi_j}")
            self.logger.debug(
                f"[Bar {i}] Local forces: L_i={l_i}, T_i={t_i}, "
                f"L_j={l_j}, T_j={t_j}")
            force_sec = np.array([
                [-(l_i * np.cos(phi_i) - t_i * np.sin(phi_i))],
                [-(t_i * np.cos(phi_i) + l_i * np.sin(phi_i))],
                [force[2, 0]],
                [l_j * np.cos(phi_j) - t_j * np.sin(phi_j)],
                [t_j * np.cos(phi_j) + l_j * np.sin(phi_j)],
                [force[5, 0]]
            ])
            self.logger.debug(
                f"[Bar {i}] Transformed force vector:\n{force_sec}")
            forces_list.append(force_sec)
        self.logger.debug("Internal force transformation completed.")
        return forces_list

    def _update_geometry(self, system, node_deform_curr, node_deform_prev):
        """Apply incremental node displacements and update system geometry.

        The nodal coordinates are updated based on the incremental displacement
        difference between two iteration steps. The maximum nodal shift is also
        evaluated to check convergence.

        Parameters
        ----------
        system : :any:`System`
            Current system geometry.
        node_deform_curr : np.ndarray
            Global nodal displacement vector (size = num_nodes * dof)
        node_deform_prev : np.ndarray
            Global nodal displacement vector from previous iteration.

        Returns
        -------
        :any:`System`
            Updated system with displaced node coordinates.
        float
            Maximum nodal shift magnitude observed during this update.
        """
        self.logger.info(
            "Updating system geometry based on incremental deformations.")

        bars_new = []
        max_deform = 0.0
        node_map = {}
        nodes = system.nodes()
        dof = 3

        def get_delta(i):
            u_curr = node_deform_curr[i * dof:(i + 1) * dof]
            u_prev = node_deform_prev[i * dof:(i + 1) * dof]
            return u_curr - u_prev

        for i_bar, bar in enumerate(system.bars):

            i = nodes.index(bar.node_i)
            j = nodes.index(bar.node_j)

            delta_i = get_delta(i)
            delta_j = get_delta(j)

            deform_x_i, deform_z_i = float(delta_i[0]), float(delta_i[1])
            deform_x_j, deform_z_j = float(delta_j[0]), float(delta_j[1])

            max_deform = max(
                max_deform,
                np.hypot(deform_x_i, deform_z_i),
                np.hypot(deform_x_j, deform_z_j)
            )

            self.logger.info(f"[Bar {i_bar}] Updating geometry.")
            self.logger.debug(
                f"[Bar {i_bar}] Δu_i=({deform_x_i}, {deform_z_i}), "
                f"Δu_j=({deform_x_j}, {deform_z_j})"
            )

            if bar.node_i not in node_map:
                node_map[bar.node_i] = replace(
                    bar.node_i,
                    x=bar.node_i.x + deform_x_i,
                    z=bar.node_i.z + deform_z_i
                )
            ni = node_map[bar.node_i]

            if bar.node_j not in node_map:
                node_map[bar.node_j] = replace(
                    bar.node_j,
                    x=bar.node_j.x + deform_x_j,
                    z=bar.node_j.z + deform_z_j
                )
            nj = node_map[bar.node_j]

            self.logger.debug(
                f"[Bar {i_bar}] Updated node_i: ({ni.x}, {ni.z})")
            self.logger.debug(
                f"[Bar {i_bar}] Updated node_j: ({nj.x}, {nj.z})")

            bars_new.append(replace(bar, node_i=ni, node_j=nj))

        self.logger.info(
            f"Geometry update completed. Max nodal shift={max_deform}")
        return replace(system, bars=bars_new), max_deform

    def _run_iteration(
            self, iterations: int, tolerance: float,
            result_type: Literal['incremental', 'cumulative']
    ):
        """Perform iterative geometry updates to capture nonlinear effects.

        Runs a fixed or convergence-controlled iteration loop that successively
        updates the system geometry until convergence or the iteration limit is
        reached.

        Parameters
        ----------
        iterations : int, default=10
            Maximum number of iterations to perform.
        tolerance : float, default=1e-3
            Convergence criterion for the maximum nodal shift.

        Returns
        -------
        list of dict
            List containing iteration index, system state, and maximum nodal
            shift.
        """
        self.logger.info(
            f"Starting nonlinear iteration process: iterations={iterations}, "
            f"tolerance={tolerance}, mode='{result_type}'."
        )
        if not isinstance(iterations, int):
            msg_1 = (f'The number of iterations must be of type int. '
                     f'The current type is "{type(iterations)}".')
            self.logger.error(msg_1)
            raise ValueError(msg_1)

        if iterations <= 0:
            msg_2 = 'The number of iterations has to be greater than zero.'
            self.logger.error(msg_2)
            raise ValueError(msg_2)

        if result_type not in ('cumulative', 'incremental'):
            msg_3 = (f'Iteration mode has to be either "cumulative" or '
                     f'"incremental". Got "{result_type}" instead.')
            self.logger.error(msg_3)
            raise ValueError(msg_3)

        results: List[dict] = []
        self.logger.debug(
            "Preparing initial systems: system_prev (loads removed) "
            "and system_curr (original).")
        system_prev = SystemModifier(self.system).delete_loads()
        system_curr = self.system

        for i in range(iterations):
            self.logger.info(f"--- Iteration {i} ---")
            solver_prev = Solver(system_prev, debug=self.debug)
            solver_curr = Solver(system_curr, debug=self.debug)
            self.logger.debug("Generated solver instances for "
                              "previous and current systems.")

            node_deform_prev = solver_prev.node_deform
            node_deform_curr = solver_curr.node_deform

            system_next, max_shift = self._update_geometry(
                system_curr,
                node_deform_curr,
                node_deform_prev
            )

            entry: dict = {
                "iteration": i,
                "system": system_curr,
                "max_shift": max_shift
            }

            if result_type == 'incremental':
                self.logger.info(
                    f"[Iteration {i}] Computing incremental differences.")
                diffs = self._incremental_difference(solver_prev, solver_curr)
                entry.update(diffs)

            results.append(entry)

            if max_shift < tolerance:
                self.logger.info(f"Convergence reached at iteration {i}.")
                break

            system_prev, system_curr = system_curr, system_next

        self.logger.info("Iteration process completed.")
        self._iteration_results = results
        return results

    def _incremental_difference(self, prev, curr):
        r"""
        Compute incremental differences between two solver states.

        This helper extracts the difference of deformation vectors, support
        reactions and internal force distributions between two consecutive
        iteration steps.

        Parameters
        ----------
        prev : `Solver`
            `Solver` instance representing the previous iteration.
        curr : `Solver`
            `Solver` instance representing the current iteration.

        Returns
        -------
        dict
            A dictionary containing incremental differences for:

            - ``internal_forces_diff``
            - ``node_support_diff``
            - ``system_support_diff``
            - ``bar_deform_diff``
            - ``node_deform_diff``

            Each entry is either a list of elementwise differences or a
            vector difference if the underlying data type is not a list.

        Notes
        -----
        Differences are computed as ``curr - prev``. Lists are processed
        element-wise, other numeric structures are subtracted directly.
        """
        self.logger.debug(
            "Computing incremental differences between iterations.")

        def diff(a, b):
            if isinstance(a, list):
                return [x - y for x, y in zip(a, b)]
            else:
                return a - b

        return {
            "internal_forces_diff": diff(curr.internal_forces,
                                         prev.internal_forces),
            "node_support_forces_diff": diff(curr.node_support_forces,
                                             prev.node_support_forces),
            "system_support_forces_diff": diff(curr.system_support_forces,
                                               prev.system_support_forces),
            "bar_deform_list_diff": diff(curr.bar_deform_list,
                                         prev.bar_deform_list),
            "node_deform_diff": diff(curr.node_deform, prev.node_deform),
        }

    def _validation_approach_index(self, approach, iteration_index):
        """Validate the calculation approach and the corresponding iteration
        index.

        This method ensures that the specified solution approach and the
        provided iteration index are consistent. It verifies that the approach
        is one of the supported types and checks that an iteration index is
        supplied only when required. Invalid combinations or unsupported values
         trigger detailed error messages and exceptions.

        Parameters
        ----------
        approach : str
            The calculation approach to be used. Must be either
            ``"matrix"`` or ``"iterative"``.
        iteration_index : int or None
            The iteration index associated with the iterative approach.
            Must be ``None`` when using the matrix approach. Must be an ``int``
            and not ``None`` when the iterative approach is selected.

        Raises
        ------
        ValueError
            If an unsupported approach is provided, if an iteration index is
            given for the matrix approach, or if no valid integer iteration
            index is provided for the iterative approach.
        """
        self.logger.debug(f"[Validation] Checking approach={approach}, "
                          f"iteration_index={iteration_index}")
        if approach not in ['matrix', 'iterative']:
            msg_1 = (f'Approach has to be either "matrix" or "iterative". '
                     f'Got "{approach}" instead.')
            self.logger.error(msg_1)
            raise ValueError(msg_1)

        if approach == 'matrix' and iteration_index is not None:
            msg_2 = 'Matrix approach cannot have an iteration index.'
            self.logger.error(msg_2)
            raise ValueError(msg_2)
        if approach == 'iterative' and (iteration_index is None or
                                        not isinstance(iteration_index, int)):
            msg_3 = (f'The Iteration Index has to be an int and not None. '
                     f'Got {type(iteration_index)} instead.')
            self.logger.error(msg_3)
            raise ValueError(msg_3)

    def differential_equation(
            self, approach: Literal['matrix', 'iterative'],
            iteration_index: Optional[int] = None,
            bar_index: Optional[int] = None,
            n_disc: int = 10
    ):
        r"""Construct differential equation objects for bar deformation
        analysis.

        Depending on the chosen approach, this method generates instances of
        :class:`DifferentialEquationSecond` (matrix-based approach) or generic
        :class:`DifferentialEquation` instantiated through
        :func:`get_differential_equation` (iterative approach).

        Parameters
        ----------
        approach : {'matrix', 'iterative'}
            Defines which second-order analysis results are used.
        iteration_index : int, optional
            Required for the iterative approach. Determines which iteration
            step is used. Negative indexing is supported.
        bar_index : int, optional
            If provided, only the specified bar is processed. Negative indices
            are allowed. If omitted, all bars are processed.
        n_disc : int, default=10
            Number of subdivisions used when constructing differential
            equation functions.

        Returns
        -------
        DifferentialEquation or list of DifferentialEquation
            One object per bar or a list for all bars.

        Raises
        ------
        ValueError
            If the approach or iteration index is invalid.
        AttributeError
            If the matrix approach was requested before running
            :meth:`matrix_approach`.
        RuntimeError
            If the iterative approach was requested but no iterations exist.

        Notes
        -----
        - In cumulative mode, deformation and forces are taken directly from
            the solver of the chosen iteration.
        - In incremental mode, differential quantities from the stored
          iteration history are used.
        """
        self._validation_approach_index(approach, iteration_index)

        if approach == 'matrix':
            bars = self._modified_system_matrix.mesh.bars
            solver = self.solver_matrix_approach
            if bar_index is not None:
                return DifferentialEquationSecond(
                    bar=bars[bar_index],
                    deform=solver.bar_deform_list[bar_index],
                    forces=solver.internal_forces[bar_index],
                    n_disc=n_disc,
                    f_axial=self.averaged_longitudinal_force[bar_index]
                )
            else:
                return [
                    DifferentialEquationSecond(
                        bar=bars[i],
                        deform=solver.bar_deform_list[i],
                        forces=solver.internal_forces[i],
                        n_disc=n_disc,
                        f_axial=self.averaged_longitudinal_force[i]
                    ) for i in range(len(bars))
                ]
        else:
            if self._iteration_mode == 'cumulative':
                solver = self.solver_iteration_cumulative(iteration_index)
                return get_differential_equation(
                    self.system,
                    solver.bar_deform_list,
                    solver.internal_forces,
                    bar_index, n_disc
                )
            else:
                entry = self._iteration_results[iteration_index]
                return get_differential_equation(
                    self.system,
                    entry.get('bar_deform_diff'),
                    entry.get('internal_forces_diff'),
                    bar_index, n_disc
                )

    def plot(
            self, approach: Literal['matrix', 'iterative'],
            iteration_index: Optional[int] = None,
            kind: Literal[
                'normal', 'shear', 'moment', 'u', 'w', 'phi'] = 'normal',
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            result_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: Optional[int] = None, n_disc: int = 10
    ):
        r"""Plot second-order internal forces or deformation results.

        This method constructs a :class:`SystemResult` object based on
        either:

        - matrix-based second-order theory, or
        - (cumulative / incremental) iterative results,

        and visualizes the selected quantity using
        :class:`ResultGraphic`.

        Parameters
        ----------
        approach : {'matrix', 'iterative'}
            Defines whether the matrix-based or iterative results should
            be plotted.
        iteration_index : int, optional
            Required for the iterative approach. Specifies the iteration
            step.
            Supports negative indices.
        kind : {'normal', 'shear', 'moment', 'u', 'w', 'phi'},
                default='normal'
            Selects the result quantity to display.
        bar_mesh_type : {'bars', 'user_mesh', 'mesh'}, default='bars'
            Mesh used for the graphic bar geometry.
        result_mesh_type : {'bars', 'user_mesh', 'mesh'}, default='mesh'
            Mesh used for plotting the result distribution.
        decimals : int, optional
            Number of decimals for label annotation.
        n_disc : int, default=10
            Number of subdivisions for result interpolation.

        Raises
        ------
        ValueError
            If the approach or iteration index is invalid.
        AttributeError
            If the matrix-based approach was not initialized.
        RuntimeError
            If iterative results were requested but none exist.

        Notes
        -----
            Incremental mode plots incremental quantities (differences
            between consecutive iterations), whereas cumulative mode
            displays the absolute state of the structure at that iteration.
        """
        self._validation_approach_index(approach, iteration_index)
        from sstatics.graphic_objects import ResultGraphic
        from sstatics.core.postprocessing import SystemResult

        if approach == 'matrix':
            solver = self.solver_matrix_approach
            result = SystemResult(
                self._modified_system_matrix,
                solver.bar_deform_list,
                solver.internal_forces,
                solver.node_deform,
                solver.node_support_forces,
                solver.system_support_forces,
                n_disc=n_disc
            )
        else:
            if self._iteration_mode == 'cumulative':
                solver = self.solver_iteration_cumulative(iteration_index)
                result = SystemResult(
                    self.system,
                    solver.bar_deform_list,
                    solver.internal_forces,
                    solver.node_deform,
                    solver.node_support_forces,
                    solver.system_support_forces,
                    n_disc=n_disc
                )
            else:
                entry = self._iteration_results[iteration_index]
                result = SystemResult(
                    self.system,
                    entry['bar_deform_list_diff'],
                    entry['internal_forces_diff'],
                    entry['node_deform_diff'],
                    entry['node_support_forces_diff'],
                    entry['system_support_forces_diff'],
                    n_disc=n_disc
                )

        ResultGraphic(result, kind, bar_mesh_type, result_mesh_type,
                      decimals).show()
