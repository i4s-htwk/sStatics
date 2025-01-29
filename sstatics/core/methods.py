
from dataclasses import dataclass, replace
from functools import cached_property, cache
from typing import Literal

import numpy as np

from sstatics.core import BarTemp, System


@dataclass(eq=False)
class FirstOrder:

    system: System

    def __post_init__(self):
        self.dof = 3
        self.order = 'first'
        self.approach = None

    def _get_zero_matrix(self):
        x = len(self.system.nodes()) * self.dof
        return np.zeros((x, x))

    def _get_zero_vec(self):
        x = len(self.system.nodes()) * self.dof
        return np.zeros((x, 1))

    def _get_f_axial(self, index):
        if self.order == 'second':
            return self.averaged_longitudinal_force[index]
        return 0

    @cache
    def stiffness_matrix(self):
        k_system = self._get_zero_matrix()
        nodes = self.system.nodes()
        for index, bar in enumerate(self.system.segmented_bars):
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            f_axial = self._get_f_axial(index)

            k = bar.stiffness_matrix(
                self.order, self.approach, f_axial=f_axial)

            k_system[i:i + self.dof, i:i + self.dof] += k[:self.dof, :self.dof]
            k_system[i:i + self.dof, j:j + self.dof] += (
                k[:self.dof, self.dof:2 * self.dof])
            k_system[j:j + self.dof, i:i + self.dof] += (
                k[self.dof:2 * self.dof, :self.dof])
            k_system[j:j + self.dof, j:j + self.dof] += (
                k[self.dof:2 * self.dof, self.dof:2 * self.dof])
        return k_system

    @cache
    def elastic_matrix(self):
        elastic = self._get_zero_matrix()
        nodes = self.system.nodes()
        for bar in self.system.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            el_bar = np.vstack((
                np.hstack((bar.node_i.elastic_support, np.zeros((3, 3)))),
                np.hstack((np.zeros((3, 3)), bar.node_j.elastic_support))
            ))

            elastic[i:i + self.dof, i:i + self.dof] = (
                el_bar)[:self.dof, :self.dof]
            elastic[i:i + self.dof, j:j + self.dof] = (
                el_bar[:self.dof, self.dof:2 * self.dof])
            elastic[j:j + self.dof, i:i + self.dof] = (
                el_bar[self.dof:2 * self.dof, :self.dof])
            elastic[j:j + self.dof, j:j + self.dof] = (
                el_bar[self.dof:2 * self.dof, self.dof:2 * self.dof])
        return elastic

    @cache
    def system_matrix(self):
        return self.stiffness_matrix() + self.elastic_matrix()

    @cache
    def f0(self):
        f0_system = self._get_zero_vec()
        nodes = self.system.nodes()
        for index, bar in enumerate(self.system.segmented_bars):
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            f_axial = self._get_f_axial(index)

            f0 = bar.f0(self.order, self.approach, f_axial=f_axial)

            f0_system[i:i + self.dof, :] += f0[:self.dof, :]
            f0_system[j:j + self.dof, :] += f0[self.dof:2 * self.dof, :]
        return f0_system

    @cache
    def p0(self):
        p0 = self._get_zero_vec()
        for i, node in enumerate(self.system.nodes()):
            p0[i * self.dof:i * self.dof + self.dof, :] = (
                node.load)
        return p0

    @cache
    def p(self):
        return self.p0() - self.f0()

    @cache
    def apply_boundary_conditions(self):
        k = self.system_matrix()
        p = self.p()
        for idx, node in enumerate(self.system.nodes()):
            node_offset = idx * self.dof
            for dof_nr, attribute in enumerate(['u', 'w', 'phi']):
                condition = getattr(node, attribute, 'free')
                if condition == 'fixed':
                    k[node_offset + dof_nr, :] = 0
                    k[:, node_offset + dof_nr] = 0
                    k[node_offset + dof_nr, node_offset + dof_nr] = 1
                    p[node_offset + dof_nr] = 0
        return k, p

    @cache
    def node_deformation(self):
        modified_stiffness_matrix, modified_p = (
            self.apply_boundary_conditions())
        return np.linalg.solve(modified_stiffness_matrix, modified_p)

    def create_list_node_deformation(self):
        node_deform = self.node_deformation()
        nodes = self.system.nodes()
        return [
            np.vstack([
                node_deform[nodes.index(bar.node_i) *
                            self.dof: nodes.index(bar.node_i) * self.dof +
                            self.dof, :],
                node_deform[nodes.index(bar.node_j) *
                            self.dof: nodes.index(bar.node_j) * self.dof +
                            self.dof, :]
            ])
            for bar in self.system.segmented_bars
        ]

    @cache
    def bar_deform(self):
        node_deform = self.node_deformation()
        nodes = self.system.nodes()
        return [
            np.transpose(bar.transformation_matrix())
            @ np.vstack([
                node_deform[nodes.index(bar.node_i) *
                            self.dof: nodes.index(bar.node_i) * self.dof +
                            self.dof, :],
                node_deform[nodes.index(bar.node_j) *
                            self.dof: nodes.index(bar.node_j) * self.dof +
                            self.dof, :]
            ])
            for bar in self.system.segmented_bars
        ]

    @cache
    def create_list_of_bar_forces(self):
        bar_deform = self.bar_deform()
        return [
            bar.stiffness_matrix(to_node_coord=False,
                                 f_axial=self._get_f_axial(i)) @ deform +
            bar.f0(to_node_coord=False,
                   f_axial=self._get_f_axial(i)) + bar.f0_point
            for i, (bar, deform) in
            enumerate(zip(self.system.segmented_bars, bar_deform))
        ]

    @cache
    def _apply_hinge_modification(self):
        deform_list = []
        bar_deform_list = self.bar_deform()
        for i, bar in enumerate(self.system.segmented_bars):
            delta_rel = np.zeros((6, 1))
            if True in bar.hinge:
                k = bar.stiffness_matrix(
                    self.order, self.approach, hinge_modification=False,
                    f_axial=self._get_f_axial(i)
                )
                bar_deform = bar_deform_list[i]
                f0 = bar.f0(
                    self.order, self.approach, hinge_modification=False,
                    f_axial=self._get_f_axial(i))

                idx = [i for i, value in enumerate(bar.hinge) if value]
                if idx:
                    rhs = (-np.dot(k[np.ix_(idx, range(6))], bar_deform)
                           - f0[idx])
                    delta_rel_reduced = (
                        np.linalg.solve(k[np.ix_(idx, idx)], rhs)
                    )
                    delta_rel[idx] = delta_rel_reduced
            deform_list.append(delta_rel)
        return deform_list

    @cache
    def bar_deform_node_displacement(self):
        return [
            np.transpose(bar.transformation_matrix())
            @ np.vstack(
                (bar.node_i.displacement, bar.node_j.displacement))
            for bar in self.system.segmented_bars
        ]

    @cache
    def create_bar_deform_list(self):
        hinge_modifications = self._apply_hinge_modification()
        bar_deforms = self.bar_deform()
        bar_deform_node_displacement = self.bar_deform_node_displacement()
        combined_results = []
        for i in range(len(hinge_modifications)):
            result = (hinge_modifications[i] + bar_deforms[i] +
                      bar_deform_node_displacement[i])
            combined_results.append(result)
        return combined_results

    @cache
    def solvable(self):
        k, p = self.apply_boundary_conditions()
        if np.linalg.matrix_rank(k) < k.shape[0]:
            print("Stiffness matrix is singular.")
            return False
        return True

    @cache
    def calc(self):
        if self.solvable():
            return (
                self.create_bar_deform_list(),
                self.create_list_of_bar_forces())

    @cached_property
    def averaged_longitudinal_force(self):
        l_averaged_list = []
        original_order = self.order
        self.order = 'first'

        for deform, force in zip(
                self.create_bar_deform_list(),
                self.create_list_of_bar_forces()):
            phi_i, phi_j = deform[2, 0], deform[5, 0]
            n_i, n_j = -force[0, 0], force[3, 0]
            v_i, v_j = -force[1, 0], force[4, 0]

            l_i = n_i * np.cos(phi_i) + v_i * np.sin(phi_i)
            l_j = n_j * np.cos(phi_j) + v_j * np.sin(phi_j)
            l_averaged = (l_i + l_j) / 2
            l_averaged_list.append(l_averaged)

        self.order = original_order
        return l_averaged_list


# TODO: Idee besprechen
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
        if self.solvable():
            if self.approach == 'iterativ':
                iteration_data = []
                iteration_data.append(self.recursive_iteration(
                    self.system, self.initial_system_iteration,
                    0, self.iterations, self.iteration_tolerance,
                    iteration_data, self.iteration_type,
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.segmented_bars))],
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.segmented_bars))],
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.segmented_bars))],
                    [np.zeros((6, 1)) for _ in range(
                        len(self.system.segmented_bars))]))
                return iteration_data
            else:
                return (self.create_bar_deform_list(),
                        self._conversion_transversial_in_iternal_force)

    @cached_property
    def _conversion_transversial_in_iternal_force(self):
        forces_list = []
        for deform, force in zip(
                self.create_bar_deform_list(),
                self.create_list_of_bar_forces()):
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
            input_system).create_list_node_deformation()
        node_deform_previous = FirstOrder(
            previous_system).create_list_node_deformation()

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
                input_system).create_bar_deform_list()
            bar_forces_current = FirstOrder(
                input_system).create_list_of_bar_forces()

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
                    input_system).create_bar_deform_list(),
                'node_displacement': node_deform_current,
                'internal_forces': FirstOrder(
                    input_system).create_list_of_bar_forces()
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
