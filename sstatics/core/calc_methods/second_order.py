
from dataclasses import dataclass, replace, field, fields
from functools import cached_property
import numpy as np
from typing import Literal
from sstatics.core.logger_mixin import LoggerMixin

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

    Notes
    -----
        **Matrix-based approach**

        This method modifies the element stiffness matrix and the member load
        vector to account for second-order effects
        (see class :any:`BarSecond`). It supports the
        following formulations:

        - **Analytical solution**
        - **Taylor series expansion**
        - **P–Δ (P-Delta) effect**

        The corresponding solver objects can be accessed through
        :py:attr:`solver_analytic`, :py:attr:`solver_taylor`
        and :py:attr:`solver_p_delta`. After initialization, each solver
        provides methods to evaluate and retrieve results.

        **Iterative approach**

        This method captures geometric nonlinearity through successive updates
        of the nodal displacements. The system geometry is iteratively
        recalculated until convergence is reached or the maximum number of
        iterations is exceeded.
    """
    system: System
    debug: bool = False

    _analytic_system: System = field(default=None, init=False, repr=False)
    _p_delta_system: System = field(default=None, init=False, repr=False)
    _taylor_system: System = field(default=None, init=False, repr=False)
    _iteration_results: list = field(default=None, init=False, repr=False)

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

    @staticmethod
    def _solver_difference(solver_a, solver_b):
        """Compute difference between two solver states.

        Compares numerical attributes of two solvers (arrays, scalars, or lists
        of arrays) and returns a solver representing the incremental
        difference.

        Parameters
        ----------
        solver_a, solver_b : :class:`Solver`
            Solvers to compare. Both must be based on compatible systems.

        Returns
        -------
        :class:`Solver`
            Solver object containing the element-wise or numerical differences
            between the two input solvers.
        """
        diff_solver = solver_b

        for key, value_b in vars(solver_b).items():
            if key.startswith('_'):
                continue
            if not hasattr(solver_a, key):
                continue

            value_a = getattr(solver_a, key)
            value = value_b

            try:
                if isinstance(value_b, np.ndarray) and isinstance(value_a,
                                                                  np.ndarray):
                    value = value_b - value_a
                elif np.isscalar(value_b) and np.isscalar(value_a):
                    value = value_b - value_a
                elif isinstance(value_b, list) and all(
                        isinstance(v, np.ndarray) for v in value_b):
                    value = [b - a for a, b in zip(value_a, value_b)]
            except Exception:
                pass

            object.__setattr__(diff_solver, key, value)

        return diff_solver

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

    def get_analytic_system(self):
        """Return the system computed by the analytic second-order solver.

        Returns
        -------
        :any:`System`
            The analytically computed second-order system.

        Warns
        -----
        UserWarning
            If the analytic system has not been computed yet.
        """
        if self._analytic_system is None:
            self.logger.warning(
                "Analytic system not yet computed. "
                "Run solver_analytic first.")
            raise ValueError
        return self._analytic_system

    def get_p_delta_system(self):
        """Return the system computed by the P–Δ (P-Delta) solver.

        Returns
        -------
        :any:`System`
            The system considering P–Δ effects.

        Raises
        ------
        UserWarning
            If the P–Δ system has not been computed yet.
        """
        if self._p_delta_system is None:
            self.logger.warning(
                "Analytic system not yet computed. "
                "Run solver_analytic first.")
            raise ValueError
        return self._p_delta_system

    def get_taylor_system(self):
        """Return the system computed by the Taylor series solver.

        Returns
        -------
        :any:`System`
            The system based on the Taylor series second-order approach.

        Raises
        ------
        UserWarning
            If the Taylor system has not been computed yet.
        """
        if self._taylor_system is None:
            self.logger.warning(
                "Analytic system not yet computed. "
                "Run solver_analytic first.")
            raise ValueError
        return self._taylor_system

    def get_iteration_system(self, iteration: int = -1):
        """Return the system geometry from a specific iteration step.

        By default, the last iteration system is returned.
        Negative indices follow standard Python indexing, e.g., ``-1``
        refers to the last computed iteration.

        Parameters
        ----------
        iteration : int, default=-1
            Iteration index to retrieve. ``-1`` corresponds to the latest
            iteration.

        Returns
        -------
        :any:`System`
            System state from the specified iteration.

        Raises
        ------
        RuntimeError
            If no iterations have been performed.
        ValueError
            If the requested iteration index is not found.
        """
        if self._iteration_results is None:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run get_iteration_results() first."
            )
        try:
            return self.get_iteration_results()[iteration]["system"]
        except IndexError:
            raise ValueError(f"Iteration {iteration} not found.")

    @cached_property
    def solver_analytic(self):
        """Compute and return the analytic second-order solver.

        Creates a solver instance using the analytic matrix-based formulation
        and stores its associated system internally.

        Returns
        -------
        :class:`Solver`
            Solver configured for analytic second-order analysis.
        """
        self._analytic_system = System(self._convert_bars('analytic'))
        solver = Solver(self._analytic_system)
        object.__setattr__(solver, 'internal_forces',
                           self._transform_internal_forces(solver))
        return solver

    @cached_property
    def solver_p_delta(self):
        """Compute and return the P–Δ (P-Delta) second-order solver.

        Creates a solver instance based on the P–Δ approach and stores its
        associated system internally.

        Returns
        -------
        :class:`Solver`
            Solver configured for P–Δ second-order analysis.
        """
        self._p_delta_system = System(self._convert_bars('p_delta'))
        solver = Solver(self._p_delta_system)
        object.__setattr__(solver, 'internal_forces',
                           self._transform_internal_forces(solver))
        return solver

    @cached_property
    def solver_taylor(self):
        """Compute and return the Taylor series second-order solver.

        Creates a solver instance using a Taylor expansion of nonlinear terms
        and stores its associated system internally.

        Returns
        -------
        :class:`Solver`
            Solver configured for Taylor series second-order analysis.
        """
        self._taylor_system = System(self._convert_bars('taylor'))
        solver = Solver(self._taylor_system)
        object.__setattr__(solver, 'internal_forces',
                           self._transform_internal_forces(solver))
        return solver

    def solver_iteration_cumulativ(self, iteration: int = -1):
        """Return a solver instance for a specific iteration step.

        By default, the last iteration solver is returned.
        Negative indices follow standard Python indexing, e.g., ``-1``
        refers to the last computed iteration.

        Parameters
        ----------
        iteration : int, default=-1
            Iteration index to generate a solver for. ``-1`` corresponds to the
            latest iteration.

        Returns
        -------
        :class:`Solver`
            Solver for the specified iteration step.

        Raises
        ------
        RuntimeError
            If no iteration results are available.
        ValueError
            If the specified iteration index does not exist.
        """
        if self._iteration_results is None:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run get_iteration_results() first."
            )
        try:
            system = self.get_iteration_results()[iteration]["system"]
            return Solver(system)
        except IndexError:
            raise ValueError(f"Iteration {iteration} not found.")

    def _run_iteration(self, iterations: int = 10, tolerance: float = 1e-3):
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
        results = []
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

            results.append({
                "iteration": i,
                "system": system_curr,
                "max_shift": max_shift
            })

            if max_shift < tolerance:
                break

            system_prev, system_curr = system_curr, system_next

        self._iteration_results = results
        return results

    def get_iteration_results(
            self, iterations: int = 10, tolerance: float = 1e-3):
        """Return stored iteration results or start computation if none exist.

        Parameters
        ----------
        iterations : int, default=10
            Maximum number of iterations if computation is triggered.
        tolerance : float, default=1e-3
            Convergence tolerance for the iteration process.

        Returns
        -------
        list of dict
            Stored or newly computed iteration results.
        """
        if self._iteration_results is None:
            print("No iteration results found — starting computation.")
            return self._run_iteration(iterations, tolerance)
        return self._iteration_results

    @property
    def iteration_count(self):
        """Return the number of performed iterations.

        Returns
        -------
        int
            Number of iteration steps computed.

        Raises
        ------
        RuntimeError
            If no iterations have been performed yet.
        """
        if self._iteration_results is None:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run get_iteration_results() first."
            )
        return len(self.get_iteration_results())

    def solver_iteration_incremental(self, iteration_index: int = -1):
        """Return the incremental solver between two successive iterations.

        The resulting solver represents the difference between iteration
        `iteration_index-1` and `iteration_index`.

        Parameters
        ----------
        iteration_index : int, default=-1
            Index of the current iteration. ``-1`` corresponds to the
            latest iteration.

        Returns
        -------
        :class:`Solver`
            Solver containing incremental internal forces and deformations.

        Raises
        ------
        RuntimeError
            If no iteration results exist.
        """
        if self._iteration_results is None:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run get_iteration_results() first."
            )

        solver_prev = Solver(
            self.get_iteration_results()[iteration_index - 1]["system"])
        solver_curr = Solver(
            self.get_iteration_results()[iteration_index]["system"])
        diff_solver = Solver(solver_curr.system)

        object.__setattr__(
            diff_solver, 'internal_forces',
            [c - p for p, c in zip(solver_prev.internal_forces,
                                   solver_curr.internal_forces)]
        )
        object.__setattr__(
            diff_solver, 'node_deform_list',
            [c - p for p, c in zip(solver_prev.node_deform_list,
                                   solver_curr.node_deform_list)]
        )
        object.__setattr__(
            diff_solver, 'bar_deform_list',
            [c - p for p, c in zip(solver_prev.bar_deform_list,
                                   solver_curr.bar_deform_list)]
        )
        object.__setattr__(
            diff_solver, 'system_deform_list',
            [c - p for p, c in zip(solver_prev.system_deform_list,
                                   solver_curr.system_deform_list)]
        )
        object.__setattr__(
            diff_solver, 'node_support_forces',
            [c - p for p, c in zip(solver_prev.node_support_forces,
                                   solver_curr.node_support_forces)]
        )
        object.__setattr__(
            diff_solver, 'system_support_forces',
            [c - p for p, c in zip(solver_prev.system_support_forces,
                                   solver_curr.system_support_forces)]
        )

        return diff_solver

    def iteration_matrix(
            self, mode: Literal['cumulative', 'incremental'] = "cumulative"):
        """Return iteration results as a structured list of solver data.

        Collects solver results from all iterations in either *cumulative* or
        *incremental* mode. The output includes internal forces, deformation
        lists and geometric shift data for each iteration.

        Parameters
        ----------
        mode : {'cumulative', 'incremental'}, default='cumulative'
            Defines how iteration results are interpreted:
            - ``'cumulative'`` : total results up to each iteration step
            - ``'incremental'`` : differences between consecutive iterations

        Returns
        -------
        list of dict
            Each dictionary contains solver data, including:
            ``internal_forces``, ``node_deform_list``, ``bar_deform_list``, and
            ``max_shift``.

        Raises
        ------
        RuntimeError
            If no iteration results are available.
        ValueError
            If `mode` is not one of {'cumulative', 'incremental'}.
        """
        if self._iteration_results is None:
            raise RuntimeError(
                "No iterations have been performed yet. "
                "Run get_iteration_results() first."
            )
        if mode not in ['cumulative', 'incremental']:
            raise ValueError(
                f'Iteration mode has to be either "cumulative" or '
                f'"incremental". '
                f'Got "{mode}" instead.'
            )
        matrix = []
        results = self.get_iteration_results()

        for i, r in enumerate(results):
            solver = (
                self.solver_iteration_cumulativ(i)
                if mode == "cumulative"
                else self.solver_iteration_incremental(i)
            )
            if solver is None:
                continue
            matrix.append({
                "iteration": r["iteration"],
                "internal_forces": solver.internal_forces,
                "node_deform_list": solver.node_deform_list,
                "bar_deform_list": solver.bar_deform_list,
                "system_deform_list": getattr(solver, "system_deform_list",
                                              None),
                "max_shift": r["max_shift"]
            })
        return matrix
