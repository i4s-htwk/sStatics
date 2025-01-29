
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np

from sstatics.core import (
    Bar, Node, System, SystemModifier
)


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

    @cache
    def stiffness_matrix(self):
        k_system = self._get_zero_matrix()
        nodes = self.system.nodes()
        for bar in self.system.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            k = bar.stiffness_matrix(self.order, self.approach)

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
        for bar in self.system.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            f0 = bar.f0(self.order, self.approach)

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
            bar.stiffness_matrix(to_node_coord=False) @ deform +
            bar.f0(to_node_coord=False) + bar.f0_point
            for bar, deform in zip(self.system.segmented_bars, bar_deform)
        ]

    @cache
    def _apply_hinge_modification(self):
        deform_list = []
        bar_deform_list = self.bar_deform()
        for i, bar in enumerate(self.system.segmented_bars):
            delta_rel = np.zeros((6, 1))
            if True in bar.hinge:
                k = bar.stiffness_matrix(self.order, self.approach,
                                         hinge_modification=False)
                bar_deform = bar_deform_list[i]
                f0 = bar.f0(self.order, self.approach,
                            hinge_modification=False)

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


@dataclass(eq=False)
class InfluenceLine:

    system: System

    def __post_init__(self):
        self.dof = 3
        self.modifier = SystemModifier(self.system)

    def force(self, force: Literal['fx', 'fz', 'fm'], obj,
              position: float = 0):
        if force not in ['fx', 'fz', 'fm']:
            raise ValueError(f"Invalid force type: {force}")

        if isinstance(obj, Bar):
            self.modified_system = self.modifier.modify_bar_force(
                obj, force, position, virt_force=1)

        elif isinstance(obj, Node):
            if position:
                raise ValueError(
                    "If obj is an instance of Node, position must be None.")
            self.modified_system = self.modifier.modify_node_force(
                obj, force, virt_force=1)
        else:
            raise ValueError("obj must be an instance of Bar or Node")

        calc_system = FirstOrder(self.modified_system)

        if calc_system.solvable():
            norm_force = self.calc_norm_force(force, obj)
            if isinstance(obj, Bar):
                self.modified_system = self.modifier.modify_bar_force(
                    obj, force, position, virt_force=norm_force)
            elif isinstance(obj, Node):
                self.modified_system = self.modifier.modify_node_force(
                    obj, force, virt_force=norm_force)
            calc_system = FirstOrder(self.modified_system)
            return calc_system.calc()
        else:
            # TODO: Hier wird die Verschiebungsfigur zurückgegeben und
            #       nicht deform, force wie bei calc_system.calc()
            return None

    def calc_norm_force(self, force: Literal['fx', 'fz', 'fm'],
                        obj):
        """
        Normalize the deformation of the bar system based on the given force.
        This method calculates a virtual force to balance the deformation
        difference between two connected bars, based on their deformation.
        """
        calc_system = FirstOrder(self.modified_system)
        if isinstance(obj, Bar):
            # calc bar deformations
            deform = calc_system.create_bar_deform_list()

            # Get the index of the bar in the system
            bars = list(self.system.bars)
            idx = bars.index(obj)

            deform_bar_i, deform_bar_j = deform[idx], deform[idx + 1]

            # Map the force type to corresponding indices for the deformation
            # values
            force_indices = {'fx': (3, 0), 'fz': (4, 1), 'fm': (5, 2)}
            idx_i, idx_j = force_indices[force]

            # Calculate the difference in deformation between the two bars
            delta = deform_bar_j[idx_j][0] - deform_bar_i[idx_i][0]
        elif isinstance(obj, Node):
            node_deformation = calc_system.node_deformation()
            for i, node in enumerate(self.system.nodes()):
                if node == obj:
                    node_deform = node_deformation[
                               i * self.dof:i * self.dof + self.dof, :]
                    force_indices = {'fx': 0, 'fz': 1, 'fm': 2}
                    delta = node_deform[force_indices[force]][0]
                    break
        else:
            raise ValueError("obj must be an instance of Bar or Node")

        if delta == 0:
            raise ZeroDivisionError("Deformation difference (delta) is zero, "
                                    "cannot calculate norm force.")
        return -1 * float(np.abs(1 / delta))

    def deform(self, deform: Literal['u', 'w', 'phi'], obj: Bar,
               position: float = 0):
        if deform not in ['u', 'w', 'phi']:
            raise ValueError(f"Invalid deform type: {deform}")

        if isinstance(obj, Bar):
            if not (0 <= position <= 1):
                raise ValueError(
                    f"Position {position} must be between 0 and 1.")

            self.modified_system = (
                self.modifier.modify_bar_deform(obj, deform, position))

            calc_system = FirstOrder(self.modified_system)

            return calc_system.calc()
        raise ValueError("obj must be an instance of Bar")
