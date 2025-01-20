
from dataclasses import dataclass
from functools import cache

import numpy as np

from sstatics.core import Bar


@dataclass(eq=False)
class System:

    bars: tuple[Bar, ...] | list[Bar]

    # weitere Validierungen? sich schneidende Stäbe?
    def __post_init__(self):
        self.bars = tuple(self.bars)
        if len(self.bars) == 0:
            raise ValueError('There need to be at least one bar.')
        for i, bar in enumerate(self.bars[0:-1]):
            if any([
                bar.same_location(other_bar) for other_bar in self.bars[i + 1:]
            ]):
                raise ValueError(
                    'Cannot instantiate a system with bars that share the '
                    'same location.'
                )
        nodes = self.nodes(segmented=False)
        for i, node in enumerate(nodes[0:-1]):
            for other_node in nodes[i + 1:]:
                if node.same_location(other_node) and node != other_node:
                    raise ValueError(
                        'Inconsistent system. Nodes with the same location '
                        'need to be the same instance.'
                    )
        to_visit, visited = [nodes[0]], []
        while to_visit:
            current_node = to_visit.pop(0)
            if current_node not in visited:
                visited.append(current_node)
                to_visit += self.connected_nodes(segmented=False)[current_node]
        if set(visited) != set(nodes):
            raise ValueError("The system's graph needs to be connected.")

        self.segmented_bars = []
        for bar in self.bars:
            self.segmented_bars += bar.segment()
        self.segmented_bars = tuple(self.segmented_bars)

        self.dof = 3

    @cache
    def connected_nodes(self, segmented: bool = True):
        bars = self.segmented_bars if segmented else self.bars
        connections = {}
        for bar in bars:
            for node in (bar.node_i, bar.node_j):
                if node not in connections:
                    connections[node] = set()
            connections[bar.node_i].add(bar.node_j)
            connections[bar.node_j].add(bar.node_i)
        return {
            node: list(connected_nodes)
            for node, connected_nodes in connections.items()
        }

    @cache
    def nodes(self, segmented: bool = True):
        return list(self.connected_nodes(segmented=segmented).keys())

    def _get_zero_matrix(self):
        x = len(self.nodes()) * self.dof
        return np.zeros((x, x))

    def _get_zero_vec(self):
        x = len(self.nodes()) * self.dof
        return np.zeros((x, 1))

    def stiffness_matrix(self, order: str = 'first',
                         approach: str | None = None):
        k_system = self._get_zero_matrix()
        nodes = self.nodes()
        for bar in self.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            k = bar.stiffness_matrix()

            k_system[i:i + self.dof, i:i + self.dof] += k[:self.dof, :self.dof]
            k_system[i:i + self.dof, j:j + self.dof] += (
                k[:self.dof, self.dof:2 * self.dof])
            k_system[j:j + self.dof, i:i + self.dof] += (
                k[self.dof:2 * self.dof, :self.dof])
            k_system[j:j + self.dof, j:j + self.dof] += (
                k[self.dof:2 * self.dof, self.dof:2 * self.dof])
        return k_system

    def elastic_matrix(self):
        elastic = self._get_zero_matrix()
        nodes = self.nodes()
        for bar in self.segmented_bars:
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

    def system_matrix(self, order: str = 'first',
                      approach: str | None = None):
        return self.stiffness_matrix(order, approach) + self.elastic_matrix()

    def f0(self, order: str = 'first', approach: str | None = None):
        f0_system = self._get_zero_vec()
        nodes = self.nodes()
        for bar in self.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            f0 = bar.f0()

            f0_system[i:i + self.dof, :] += f0[:self.dof, :]
            f0_system[j:j + self.dof, :] += f0[self.dof:2 * self.dof, :]
        return f0_system

    def p0(self):
        p0 = self._get_zero_vec()
        for i, node in enumerate(self.nodes()):
            p0[i * self.dof:i * self.dof + self.dof, :] = (
                node.load)
        return p0

    def p(self):
        return self.p0() - self.f0()

    def apply_boundary_conditions(self, order: str = 'first',
                                  approach: str | None = None):
        k = self.system_matrix(order, approach)
        p = self.p()
        for idx, node in enumerate(self.nodes()):
            node_offset = idx * self.dof
            for dof_nr, attribute in enumerate(['u', 'w', 'phi']):
                condition = getattr(node, attribute, 'free')
                if condition == 'fixed':
                    k[node_offset + dof_nr, :] = 0
                    k[:, node_offset + dof_nr] = 0
                    k[node_offset + dof_nr, node_offset + dof_nr] = 1
                    p[node_offset + dof_nr] = 0
        return k, p

    def node_deformation(self, order: str = 'first',
                         approach: str | None = None):
        modified_stiffness_matrix, modified_p = (
            self.apply_boundary_conditions(order, approach))
        return np.linalg.solve(modified_stiffness_matrix, modified_p)

    def bar_deform(self, order: str = 'first', approach: str | None = None):
        node_deform = self.node_deformation(order, approach)
        nodes = self.nodes()

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
            for bar in self.segmented_bars
        ]

    def create_list_of_bar_forces(self, order: str = 'first',
                                  approach: str | None = None):
        bar_deform = self.bar_deform(order, approach)
        f_node = [
            bar.stiffness_matrix() @ deform +
            bar.f0()
            for bar, deform in zip(self.segmented_bars, bar_deform)
        ]
        return [
            np.transpose(
                bar.transformation_matrix()) @ forces - bar.f0_point_load
            for bar, forces in zip(self.segmented_bars, f_node)
        ]

    def _apply_hinge_modification(self, order: str = 'first',
                                  approach: str | None = None):
        deform_list = []
        bar_deform_list = self.bar_deform(order, approach)
        for i, bar in enumerate(self.segmented_bars):
            delta_rel = np.zeros((6, 1))
            if True in bar.hinge:
                k = bar.stiffness_matrix(order, approach,
                                         hinge_modification=False)
                bar_deform = bar_deform_list[i]
                f0 = bar.f0(order, approach, hinge_modification=False)

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

    def bar_deform_node_displacement(self):
        return [
            np.transpose(bar.transformation_matrix())
            @ np.vstack(
                (bar.node_i.displacement, bar.node_j.displacement))
            for bar in self.segmented_bars
        ]

    def create_bar_deform_list(self, order: str = 'first',
                               approach: str | None = None):
        hinge_modifications = self._apply_hinge_modification(order, approach)
        bar_deforms = self.bar_deform(order, approach)
        bar_deform_node_displacement = self.bar_deform_node_displacement()
        combined_results = []
        for i in range(len(hinge_modifications)):
            result = (hinge_modifications[i] + bar_deforms[i] +
                      bar_deform_node_displacement[i])
            combined_results.append(result)
        return combined_results

    def solvable(self, order: str = 'first', approach: str | None = None):
        k, p = self.apply_boundary_conditions(order, approach)
        if np.linalg.matrix_rank(k) < k.shape[0]:
            print("Stiffness matrix is singular.")
            return False
        return True
