
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

    - ``"analytic"`` — closed-form analytical second-order stiffness matrix
    - ``"taylor"`` — Taylor series expansion of the exact formulation
    - ``"p_delta"`` — classical P–Δ geometric stiffness formulation

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
    """
    system: System
    debug: bool = False

    _modified_system_matrix: Optional[System] = field(init=False, default=None)
    _solution_matrix: Optional[Solver] = field(init=False, default=None)
    _iteration_results: List[dict] = field(
        default_factory=list, init=False, repr=False)
    _iteration_mode: Literal['cumulative', 'incremental'] | None = field(
        init=False, default=None)

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
              Closed-form exact geometric stiffness including axial force
              effects.
            - ``'p_delta'``
              Classical P–Δ geometric stiffness formulation.
            - ``'taylor'``
              Taylor-series expansion of the analytical stiffness.

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

        if approach not in ['analytic', 'p_delta', 'taylor']:
            raise ValueError(
                f'Matrix approach has to be either "analytic", "taylor" or '
                f'"p_delta". '
                f'Got "{approach}" instead.')

        # Reset Results
        self._solution_matrix = None
        self._modified_system_matrix = None

        # Set modified System
        self._modified_system_matrix = System(self._convert_bars(approach))

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
        if self._modified_system_matrix is None:
            raise AttributeError(
                "The modified system has not been created yet. "
                "Call `matrix_approach(...)` before accessing "
                "`solver_matrix_approach`."
            )
        if self._solution_matrix is None:
            solver = Solver(self._modified_system_matrix)
            object.__setattr__(solver, 'internal_forces',
                               self._transform_internal_forces(solver))
            self._solution_matrix = solver
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
        if self._modified_system_matrix is None:
            raise AttributeError(
                "The modified system has not been created yet. "
                "Call `matrix_approach()` before accessing "
                "`system_matrix_approach`."
            )
        return self._modified_system_matrix

    def iterative_approach(
            self, iterations: int = 10, tolerance: float = 1e-3,
            result_type: Literal['cumulative', 'incremental'] = 'cumulative'):
        r"""Execute an iterative geometric–nonlinear second-order analysis.

        This method performs repeated updates of the system geometry, internal
        forces and stiffness based on newly computed nodal displacements. The
        process continues until convergence is reached or the maximum number of
        iterations is exhausted.

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
        can be accessed via :meth:`solver_iterative_approach`,
        :meth:`system_iterative`, or :meth:`iteration_matrix`.

        Raises
        ------
        ValueError
            If `iterations` is not a positive integer.
        ValueError
            If `result_type` is invalid.
        """

        if not isinstance(iterations, int):
            raise ValueError(
                f'The number of iterations must be of type int. '
                f'The current type is "{type(iterations)}".'
            )

        if iterations <= 0:
            raise ValueError(
                'The number of iterations has to be greater than zero.'
            )

        if result_type not in ('cumulative', 'incremental'):
            raise ValueError(
                f'Iteration mode has to be either "cumulative" or '
                f'"incremental". Got "{result_type}" instead.'
            )
        # Reset Results
        self._iteration_results = []
        self._iteration_mode = None

        # Calculate Iteration Results
        self._iteration_results = self._run_iteration(
            iterations, tolerance, result_type)
        self._iteration_mode = result_type

    def solver_iterative_approach(self, iteration: int = -1):
        r"""Return a solver corresponding to a specific iteration step.

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

        if not self._iteration_results:
            raise AttributeError(
                "No iterations have been performed yet. "
                "Run iterative_approach() first."
            )
        try:
            entry = self._iteration_results[iteration]
        except IndexError:
            raise ValueError(f"Iteration {iteration} not found.")
        return Solver(entry["system"])

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

        if not self._iteration_results:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run get_iteration_results() first."
            )
        try:
            return self._iteration_results[iteration]["system"]
        except IndexError:
            raise ValueError(f"Iteration {iteration} not found.")

    @property
    def iteration_count(self):
        r"""Return the total number of performed iteration steps.

        Returns
        -------
        int
            Number of computed iterations.

        Raises
        ------
        RuntimeError
            If no iterative analysis has been performed yet.
        """
        if not self._iteration_results:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run get_iteration_results() first."
            )
        return len(self._iteration_results)

    def iteration_matrix(self):
        r"""
        Return all iteration results in structured matrix form.

        Each entry corresponds to one iteration and contains internal forces,
        nodal deformations, bar deformations, system deformation lists and
        maximum geometric shift. Depending on the iteration mode:

        - **cumulative mode** stores absolute values at each iteration
        - **incremental mode** stores differences between consecutive
            iterations

        Returns
        -------
        list of dict
            Each dictionary includes:

            - ``iteration`` – iteration index
            - ``internal_forces`` – internal bar forces
            - ``node_deform_list`` – nodal displacements
            - ``bar_deform_list`` – bar deformation states
            - ``max_shift`` – maximum displacement increment

        Raises
        ------
        RuntimeError
            If no iteration results are available.
        """

        if not self._iteration_results:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run iterative_approach() first."
            )

        matrix = []
        results = self._iteration_results

        for i, r in enumerate(results):
            if self._iteration_mode == "cumulative":
                solver = self.solver_iterative_approach(i)
                matrix.append({
                    "iteration": r.get("iteration", i),
                    "internal_forces": solver.internal_forces,
                    "node_deform_list": solver.node_deform_list,
                    "bar_deform_list": solver.bar_deform_list,
                    "max_shift": r["max_shift"]
                })
            else:
                matrix.append({
                    "iteration": r.get("iteration", i),
                    "internal_forces": r.get("internal_forces_diff"),
                    "node_deform_list": r.get("node_deform_diff"),
                    "bar_deform_list": r.get("bar_deform_diff"),
                    "max_shift": r["max_shift"]
                })
        return matrix

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
        solution = Solver(self.system)
        l_avg = []
        for deform, force in zip(solution.bar_deform_list,
                                 solution.internal_forces):
            l_i = -force[0, 0] * np.cos(deform[2, 0]) - force[1, 0] * np.sin(
                deform[2, 0])
            l_j = force[3, 0] * np.cos(deform[5, 0]) + force[4, 0] * np.sin(
                deform[5, 0])
            l_avg.append((l_i + l_j) / 2)
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
        bars = list(self.system.bars)
        bar_field_names = {f.name for f in fields(Bar)}
        return [
            BarSecond(
                **{k: getattr(bar, k) for k in bar_field_names},
                approach=approach,
                f_axial=f_axial
            )
            for bar, f_axial in zip(bars, self.averaged_longitudinal_force)
        ]

    @staticmethod
    def _transform_internal_forces(solver):
        """Rotates end forces using current bar-end rotations (phi_i, phi_j).

        The local force components are mapped with the instantaneous bar-end
        rotations to ensure second-order consistent reporting.

        Returns
        -------
        list[np.ndarray]
            List of (6×1) force vectors per bar in local bar coordinates.
        """
        forces_list = []
        for deform, force in zip(
                solver.bar_deform_list,
                solver.internal_forces):
            phi_i, phi_j = deform[2, 0], deform[5, 0]
            l_i, l_j = -force[0, 0], force[3, 0]
            t_i, t_j = -force[1, 0], force[4, 0]
            force_sec = np.array([
                [-(l_i * np.cos(phi_i) - t_i * np.sin(phi_i))],
                [-(t_i * np.cos(phi_i) + l_i * np.sin(phi_i))],
                [force[2, 0]],
                [l_j * np.cos(phi_j) - t_j * np.sin(phi_j)],
                [t_j * np.cos(phi_j) + l_j * np.sin(phi_j)],
                [force[5, 0]]
            ])
            forces_list.append(force_sec)
        return forces_list

    @staticmethod
    def _update_geometry(system, node_deform_curr, node_deform_prev):
        """Apply incremental node displacements and update system geometry.

        The nodal coordinates are updated based on the incremental displacement
        difference between two iteration steps. The maximum nodal shift is also
        evaluated to check convergence.

        Parameters
        ----------
        system : :any:`System`
            Current system geometry.
        node_deform_curr : list of np.ndarray
            (6×1) vectors representing current iteration displacements.
        node_deform_prev : list of np.ndarray
            (6×1) vectors representing previous iteration displacements.

        Returns
        -------
        :any:`System`
            Updated system with displaced node coordinates.
        float
            Maximum nodal shift magnitude observed during this update.
        """
        bars_new, max_deform = [], 0.0

        for bar, deform_curr, deform_prev in zip(system.bars,
                                                 node_deform_curr,
                                                 node_deform_prev):
            delta_deform = deform_curr - deform_prev
            deform_x_i, deform_z_i = (float(delta_deform[0]),
                                      float(delta_deform[1]))
            deform_x_j, deform_z_j = (float(delta_deform[3]),
                                      float(delta_deform[4]))

            max_deform = max(
                max_deform,
                np.hypot(deform_x_i, deform_z_i),
                np.hypot(deform_x_j, deform_z_j)
            )

            ni = replace(bar.node_i, x=bar.node_i.x + deform_x_i,
                         z=bar.node_i.z + deform_z_i)
            nj = replace(bar.node_j, x=bar.node_j.x + deform_x_j,
                         z=bar.node_j.z + deform_z_j)
            bars_new.append(replace(bar, node_i=ni, node_j=nj))

        return replace(system, bars=bars_new), max_deform

    def _run_iteration(
            self, iterations: int = 10, tolerance: float = 1e-3,
            result_type: Literal['incremental', 'cumulative'] = 'cumulative'
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
        results: List[dict] = []
        system_prev = SystemModifier(self.system).delete_loads()
        system_curr = self.system

        for i in range(iterations):
            solver_prev = Solver(system_prev)
            solver_curr = Solver(system_curr)

            node_deform_prev = solver_prev.node_deform_list
            node_deform_curr = solver_curr.node_deform_list

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
                diffs = self._incremental_difference(solver_prev, solver_curr)
                entry.update(diffs)

            results.append(entry)

            if max_shift < tolerance:
                break

            system_prev, system_curr = system_curr, system_next

        self._iteration_results = results
        return results

    @staticmethod
    def _incremental_difference(prev, curr):
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
        def diff(a, b):
            if isinstance(a, list):
                return [x - y for x, y in zip(a, b)]
            else:
                return a - b

        return {
            "internal_forces_diff": diff(curr.internal_forces,
                                         prev.internal_forces),
            "node_support_diff": diff(curr.node_support_forces,
                                      prev.node_support_forces),
            "system_support_diff": diff(curr.system_support_forces,
                                        prev.system_support_forces),
            "bar_deform_diff": diff(curr.bar_deform_list,
                                    prev.bar_deform_list),
            "node_deform_diff": diff(curr.node_deform,
                                     prev.node_deform),
        }

    @staticmethod
    def _validation_approach_index(approach, iteration_index):
        if approach not in ['matrix', 'iterative']:
            raise ValueError(
                f'Approach has to be either "matrix" or "iterative". '
                f'Got "{approach}" instead.'
            )

        if approach == 'matrix' and iteration_index is not None:
            raise ValueError(
                'Matrix approach cannot have an iteration index.'
            )

        if approach == 'iterative' and (iteration_index is None or
                                        not isinstance(iteration_index, int)):
            raise ValueError(
                f'The Iteration Index has to be an int and not None. '
                f'Got {type(iteration_index)} instead.'
            )

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
                solver = self.solver_iterative_approach(iteration_index)
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
            Incremental mode plots incremental quantities (differences between
            consecutive iterations), whereas cumulative mode displays the
            absolute state of the structure at that iteration.
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
                solver = self.solver_iterative_approach(iteration_index)
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
                    entry['bar_deform_diff'],
                    entry['internal_forces_diff'],
                    entry['node_deform_diff'],
                    entry['node_support_diff'],
                    entry['system_support_diff'],
                    n_disc=n_disc
                )

        ResultGraphic(result, kind, bar_mesh_type, result_mesh_type,
                      decimals).show()
