
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

import numpy as np

from sstatics.core.preprocessing.temperature import BarTemp
from sstatics.core.solution.first_order import FirstOrder


@dataclass(eq=False)
class SecondOrder(FirstOrder):

    calc_approach: (
            Literal['analytic', 'taylor', 'p_delta', 'iterativ'] | None) = None
    iteration_type: Literal['incremental', 'cumulativ'] | None = None
    iterations: float = 10
    iteration_tolerance: float = 1e-3

    def __post_init__(self):
        super().__post_init__()
        self.dof = 3
        self.order = 'second'
        self.approach = self.calc_approach

    def calc_second(self):
        if self.solvable:
            if self.approach == 'iterativ':
                iteration_data = []
                iteration_data.append(self.recursive_iteration(
                    self.system, self.initial_system_iteration,
                    0, self.iterations, self.iteration_tolerance,
                    iteration_data, self.iteration_type,
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.mesh))],
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.mesh))],
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.mesh))],
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.mesh))]))
                return iteration_data
            else:
                return (self.bar_deform_list,
                        self._conversion_transversial_in_iternal_force)

    @cached_property
    def _conversion_transversial_in_iternal_force(self):
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

    @property
    def initial_system_iteration(self):
        updated_bars = []
        for bar in self.system.bars:
            updated_bar = replace(
                bar,
                node_i=replace(bar.node_i, displacements=(), loads=()),
                node_j=replace(bar.node_j, displacements=(), loads=()),
                line_loads=(),
                point_loads=(),
                temp=BarTemp(temp_o=0, temp_u=0),
            )
            updated_bars.append(updated_bar)
        return replace(self.system, bars=updated_bars)

    def recursive_iteration(self, input_system, previous_system, iteration,
                            max_iterations, tolerance, iteration_results,
                            calculation_type,
                            total_deltas_bar=None, total_internal_forces=None,
                            total_deltas_node=None, total_deltas_system=None):

        if iteration >= max_iterations:
            return iteration_results

        current_system = replace(input_system)

        node_deform_current = FirstOrder(
            input_system).node_deform_list
        node_deform_previous = FirstOrder(
            previous_system).node_deform_list

        max_displacements = {'i': 0, 'j': 0}
        updated_bars = []
        for bar, node_deform, previous_node_deform in zip(
                current_system.bars, node_deform_current,
                node_deform_previous):

            delta_displacement = node_deform - previous_node_deform
            updated_node_i = replace(
                bar.node_i,
                x=bar.node_i.x + delta_displacement[0][0],
                z=bar.node_i.z + delta_displacement[1][0],
            )
            updated_node_j = replace(
                bar.node_j,
                x=bar.node_j.x + delta_displacement[3][0],
                z=bar.node_j.z + delta_displacement[4][0],
            )

            for node, indices in zip([bar.node_i, bar.node_j],
                                     [(0, 1), (3, 4)]):
                delta_x = abs(delta_displacement[indices[0]][0])
                delta_z = abs(delta_displacement[indices[1]][0])
                key = 'i' if node is bar.node_i else 'j'
                max_displacements[key] = max(max_displacements[key], np.sqrt(
                    delta_x ** 2 + delta_z ** 2))

            updated_bar = replace(bar, node_i=updated_node_i,
                                  node_j=updated_node_j)
            updated_bars.append(updated_bar)
        current_system = replace(current_system, bars=updated_bars)

        if calculation_type == 'incremental':
            bar_deform_current = FirstOrder(
                input_system).bar_deform_list
            bar_forces_current = FirstOrder(
                input_system).internal_forces

            for idx, (bar, bar_deform, node_deform, bar_forces) in enumerate(
                    zip(current_system.bars, bar_deform_current,
                        node_deform_current, bar_forces_current)):
                incremental_displacement_bar = bar_deform - total_deltas_bar[
                    idx]
                incremental_displacement_node = (node_deform -
                                                 total_deltas_node[idx])
                incremental_internal_forces = (bar_forces -
                                               total_internal_forces[idx])

                total_deltas_bar[idx] += incremental_displacement_bar
                total_deltas_node[idx] += incremental_displacement_node
                total_internal_forces[idx] += incremental_internal_forces

                result_dic = {
                    'bar_displacement': incremental_displacement_bar,
                    'node_displacement': incremental_displacement_node,
                    'internal_forces': incremental_internal_forces
                }
                iteration_results.append((iteration, result_dic))
        else:
            result_dic = {
                'bar_displacement': FirstOrder(
                    input_system).node_deform_list,
                'node_displacement': node_deform_current,
                'internal_forces': FirstOrder(
                    input_system).internal_forces
            }
            iteration_results.append((iteration, result_dic))

        if all(max_displacements[key] < tolerance for key in
               max_displacements):
            return iteration_results

        return self.recursive_iteration(
            current_system, input_system, iteration + 1, max_iterations,
            tolerance, iteration_results, calculation_type, total_deltas_bar,
            total_internal_forces, total_deltas_node, total_deltas_system
        )
