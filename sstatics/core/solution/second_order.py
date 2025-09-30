
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.solution.first_order import FirstOrder


@dataclass(eq=False)
class SecondOrder(FirstOrder):
    """
    Executes second-order (theory of 2nd order) analysis for the given system.

    Compared to :class:`FirstOrder`, this class can update the system geometry
    iteratively based on the current deformation state (*approach='iterativ'*),
    or reuse the first-order pipeline while providing second-order helpers for
    post-processing (*'analytic'*, *'taylor'*, *'p_delta'*).

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed.
    calc_approach : {'analytic','taylor','p_delta','iterativ'} or None
        Algorithm selection. Only ``'iterativ'`` triggers the geometry
        update loop. The other options reuse first-order results.
    iteration_type : {'incremental','cumulativ'} or None
        Storage mode when ``calc_approach='iterativ'``. With
        ``'incremental'``, each iteration stores deltas since the last
        iteration. With ``'cumulativ'`` (or None), each iteration stores the
        full state.
    iterations : int, default=10
        Maximum iteration count.
    iteration_tolerance : float, default=1e-3
        Convergence tolerance for nodal updates. The loop stops when the
        maximum nodal shift between two iterations is below this value.

    Attributes
    ----------
    dof : int
        Degrees of freedom per node (fixed at 3).
    order : str
        Analysis order flag ('second').
    approach : str | None
        Echo of ``calc_approach``.
    """
    calc_approach: (
            Literal['analytic', 'taylor', 'p_delta', 'iterativ'] | None
    ) = None
    iteration_type: Literal['incremental', 'cumulativ'] | None = None
    iterations: int = 10
    iteration_tolerance: float = 1e-3

    def __post_init__(self):
        super().__post_init__()
        self.dof = 3
        self.order = 'second'
        self.approach = self.calc_approach

    @property
    def calc(self):
        r"""Entry point for second-order analysis.

        Returns
        -------
        list | tuple | None
            If ``approach=='iterativ'``:
                A list of ``(iteration_index, result_dict)`` where
                ``result_dict`` contains:

                * ``'bar_displacement'`` : list of (6×1) arrays
                * ``'node_displacement'`` : list of (6×1) arrays
                * ``'system_displacement'`` : list of (6×1) arrays
                * ``'internal_forces'`` : list of (6×1) arrays

            Else:
                ``(bar_deform_list, converted_end_forces)`` from a single
                first-order solve, where ``converted_end_forces`` rotates
                end-forces with current bar-end rotations.

            Returns ``None`` if the system is unsolvable.
        """
        if not self.solvable:
            return None
        if self.approach == 'iterativ':
            return self.run_iteration()
        return (self.bar_deform_list, self._convert_end_forces())

    def run_iteration(self):
        """Runs the geometry-updating iteration loop.

        The algorithm solves the system, computes nodal deformations, updates
        the nodal coordinates by the incremental shift, and repeats until the
        maximum nodal shift falls below ``iteration_tolerance`` or the maximum
        number of iterations is reached.

        Returns
        -------
        list[tuple[int, dict]]
            Per-iteration results (see :py:meth:`calc` for keys).
        """
        results = []

        system_prev = SystemModifier(self.system).delete_loads()
        system_curr = self.system

        n = len(self.system.mesh)
        def z6(): return np.zeros((6, 1))
        total_deltas_bar = [z6() for _ in range(n)]
        total_deltas_node = [z6() for _ in range(n)]
        total_deltas_system = [z6() for _ in range(n)]
        total_internal_forces = [z6() for _ in range(n)]

        for iteration in range(self.iterations):
            node_deform_curr = FirstOrder(system_curr).node_deform_list
            node_deform_prev = FirstOrder(system_prev).node_deform_list

            system_next, max_shift = self._update_geometry(
                system_curr, node_deform_curr, node_deform_prev
            )

            if self.iteration_type == 'incremental':
                out = self._pack_incremental(
                    system_next, total_deltas_bar, total_deltas_node,
                    total_deltas_system, total_internal_forces
                )
            else:
                out = self._pack_cumulative(system_next, node_deform_curr)

            results.append((iteration, out))

            if max_shift < self.iteration_tolerance:
                break

            system_prev, system_curr = system_curr, system_next

        return results

    def _update_geometry(self, system, node_deform_curr, node_deform_prev):
        """Applies incremental node shifts and returns the updated system.

        The delta deformation is computed as
        ``node_deform_curr - node_deform_prev`` per bar. The corresponding
        node coordinates are updated in global space.

        Parameters
        ----------
        system : :any:`System`
            Current geometry state.
        node_deform_curr : list[np.ndarray]
            (6×1) vectors per bar for the current iteration.
        node_deform_prev : list[np.ndarray]
            (6×1) vectors per bar for the previous iteration.

        Returns
        -------
        :any:`System`
            Updated system with moved nodes.
        float
            Maximum Euclidean nodal shift encountered in this update.
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

    def _pack_incremental(self, system,
                          total_deltas_bar, total_deltas_node,
                          total_deltas_system, total_internal_forces
                          ):
        """Produces per-step deltas.

        Parameters
        ----------
        system : :any:`System`
            Geometry state for this iteration.
        total_deltas_bar : list[np.ndarray]
        total_deltas_node : list[np.ndarray]
        total_deltas_system : list[np.ndarray]
        total_internal_forces : list[np.ndarray]
            Running accumulators for bar, node, system displacements and
            internal forces.

        Returns
        -------
        dict
            Keys: ``'bar_displacement'``, ``'node_displacement'``,
            ``'system_displacement'``, ``'internal_forces'``.
            Values are lists of (6×1) arrays per bar, holding increments
            for this iteration relative to the previous iteration.
        """
        calc = FirstOrder(system)
        bar_deform = calc.bar_deform_list
        node_deform = calc.node_deform_list
        system_deform = calc.system_deform_list
        forces = calc.internal_forces

        inc_b, inc_n, inc_s, inc_f = [], [], [], []
        for i, (deltas_bar, deltas_node, deltas_system, delta_forces) \
                in enumerate(
                zip(bar_deform, node_deform, system_deform, forces)):
            incremental_displace_bar = deltas_bar - total_deltas_bar[i]
            incremental_displace_node = deltas_node - total_deltas_node[i]
            incremental_displace_sys = deltas_system - total_deltas_system[i]
            incremental_inter_forces = delta_forces - total_internal_forces[i]

            total_deltas_bar[i] += incremental_displace_bar
            total_deltas_node[i] += incremental_displace_node
            total_deltas_system[i] += incremental_displace_sys
            total_internal_forces[i] += incremental_inter_forces

            inc_b.append(incremental_displace_bar)
            inc_n.append(incremental_displace_node)
            inc_s.append(incremental_displace_sys)
            inc_f.append(incremental_inter_forces)

        return {
            'bar_displacement': inc_b,
            'node_displacement': inc_n,
            'system_displacement': inc_s,
            'internal_forces': inc_f,
        }

    def _pack_cumulative(self, system, node_deform_curr):
        """Produces cumulative state for this iteration.

        Parameters
        ----------
        system : :any:`System`
            Geometry state for this iteration.
        node_deform_curr : list[np.ndarray]
            Current nodal deformations per bar (6×1).

        Returns
        -------
        dict
            Keys as in :py:meth:`_pack_incremental`, but each value contains
            the full state (not an increment).
        """
        fo = FirstOrder(system)
        return {
            'bar_displacement': fo.bar_deform_list,
            'node_displacement': node_deform_curr,
            'system_displacement': fo.system_deform_list,
            'internal_forces': fo.internal_forces,
        }

    def _convert_end_forces(self):
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
                self.bar_deform_list,
                self.internal_forces):
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
