
from dataclasses import dataclass
from typing import Dict, Tuple
from functools import cached_property

import numpy as np

from sstatics.core.logger_mixin import table_matrix, LoggerMixin
from sstatics.core.preprocessing import Node, Bar
from sstatics.core.postprocessing.results import DifferentialEquation


@dataclass
class EquationOfWork(LoggerMixin):
    """
    Evaluate the mechanical work equation between two structural systems.

    The class forms all contributions of the mechanical work equation between
    a real system ``i`` and a virtual system ``j``. Bar-based terms (bending,
    shear, normal force, temperature) and node-based terms (elastic supports,
    imposed displacements) are assembled separately. Matching of bars and
    nodes is performed via rounded coordinate keys, which also supports the
    special case of missing bars in one of the systems as occurs in reduction
    schemes or the force-method.

    Parameters
    ----------
    solution_i : FirstOrder
        Postprocessing result of the real system.
    solution_j : FirstOrder
        Postprocessing result of the virtual system.
    debug : bool, optional
        Enables extended logging output. Default is ``False``.

    Notes
    -----
    The formulas assume polynomial representations of internal force per bar
    with coefficient arrays accessible via ``x_coef`` and ``z_coef``. Material
    and section properties are read through the nested attributes of each bar
    result.
    """

    # noinspection PyMissingConstructor
    def __init__(self, solution_i, solution_j, debug: bool = False):
        self.debug = debug
        self.solution_i = solution_i
        self.solution_j = solution_j
        self.logger.info('post_init')
        self.dof = 3
        self.diff_eq_i = self._index_bars_by_key(
            self.solution_i.differential_equation())
        self.diff_eq_j = self._index_bars_by_key(
            self.solution_j.differential_equation())

        self.node_result_i = self._build_node_result_index(
            self.solution_i.nodes, self.solution_i)
        self.node_result_j = self._build_node_result_index(
            self.solution_j.nodes, self.solution_j)

    @cached_property
    def work_matrix_bars(self) -> np.ndarray:
        """
        Assemble all bar-wise terms of the mechanical work equation.

        For every bar that exists in both systems a 5-component vector is
        assembled:

        ``[M, N, V, temp_const, temp_delta]``

        Missing bars in system ``j`` yield a zero row. The order of the rows
        follows the bar ordering of system ``i``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_bars, 5)`` containing moment, normal,
            shear, temperature constant and temperature gradient work
            contributions.
        """
        work_matrix_bars = np.zeros((len(self.diff_eq_i), 5))

        for i, key in enumerate(self.diff_eq_i):
            diff_eq_i_i = self.diff_eq_i.get(key)
            bar_i = diff_eq_i_i.bar
            diff_eq_j_i = self.diff_eq_j.get(key, None)
            if diff_eq_j_i is None:
                continue

            if "moment" in bar_i.deformations:
                work_matrix_bars[i, 0] += self._compute_moment_term(
                    diff_eq_i_i,
                    diff_eq_j_i,
                )
            if "normal" in bar_i.deformations:
                work_matrix_bars[i,  1] += self._compute_normal_term(
                    diff_eq_i_i,
                    diff_eq_j_i,
                )
            if "shear" in bar_i.deformations:
                work_matrix_bars[i, 2] += self._compute_shear_term(
                    diff_eq_i_i,
                    diff_eq_j_i,
                )

            work_matrix_bars[i, 3:5] += self._compute_temperature_term(
                diff_eq_i_i,
                diff_eq_j_i,
            )

        bar_matrix = table_matrix(matrix=self._sum_matrix(work_matrix_bars),
                                  column_names=[
                                      "Bar",
                                      "Moment",
                                      "Normal",
                                      "Shear",
                                      "Temperature Constant",
                                      "Temperature Delta",
                                      "Sum",
                                  ])

        self.logger.debug(f"{bar_matrix}")

        return work_matrix_bars

    @cached_property
    def work_matrix_nodes(self) -> np.ndarray:
        """
        Assemble all node-wise terms of the mechanical work equation.

        For each node present in both systems the following four-component
        vector is assembled:

        ``[elastic_F, elastic_M, disp_trans, disp_rot]``

        The method covers elastic support effects (springs in ``u``, ``w``,
        ``phi``) and imposed displacements. Missing nodes yield a zero row.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_nodes, 4)`` containing translational and
            rotational elastic-support work and the imposed-displacement
            contributions in translations and rotation.
        """
        work_matrix_nodes = np.zeros((len(self.node_result_i), 4))

        for i, key in enumerate(self.node_result_i):
            node_result_i_i = self.node_result_i.get(key)
            node_i = node_result_i_i["node"]
            node_result_j_i = self.node_result_j.get(key, None)

            if node_result_j_i is None:
                continue

            work_matrix_nodes[i, 0:2] += self._compute_elastic_support_term(
                node_i,
                node_result_i_i["deform"],
                node_result_j_i["deform"]
            )

            if node_i.displacements:
                work_matrix_nodes[i, 2:4] += self._compute_displacement_term(
                    node_i,
                    node_result_j_i["node_support"]
                )

        node_matrix = table_matrix(matrix=self._sum_matrix(work_matrix_nodes),
                                   column_names=[
                                       "Node",
                                       "Elastic Support F",
                                       "Elastic Support M",
                                       "Displacements u and w",
                                       "Displacements phi",
                                       "Sum",
                                   ])

        self.logger.debug(f"{node_matrix}")
        return work_matrix_nodes

    @cached_property
    def delta_ij(self) -> float:
        """
        Total work interaction between system ``i`` and system ``j``.

        This value corresponds to the scalar work term

        ``δ_ij = sum(bar terms) + sum(node terms)``

        and appears for example in reduction methods and the force method.

        Returns
        -------
        float
            The complete mechanical work contribution between both systems.
        """
        return float(
            np.sum(self.work_matrix_bars)
            + np.sum(self.work_matrix_nodes)
        )

    def _build_node_result_index(self, nodes, solution):
        """
        Build a coordinate-based index for all nodal results of one system.

        Each node is identified by its rounded coordinates. The stored values
        include the node object itself, its deformation vector and the
        reaction forces in the node cooridante system.

        Parameters
        ----------
        nodes : list of Node
            Node-objects of the system.
        solution : SystemResult
            Result-object providing ``node_deform`` and reaction vectors.

        Returns
        -------
        dict
            Mapping ``(x, z) -> {"node": node, "deform": …,
            "node_support": …}``.
        """
        arrays = {
            "node": nodes,
            "deform": solution.node_deform.reshape(-1, 3),
            "node_support": solution.node_support_forces.reshape(-1, 3),
        }

        return {
            (self._key_node(node)):
                {key: arr[i] for key, arr in arrays.items()}
            for i, node in enumerate(nodes)
        }

    def _index_bars_by_key(self, bars) -> Dict:
        """
        Build a coordinate-based index for all bar results.

        Bars are matched via rounded end-node coordinates, enabling correct
        identification of corresponding bars even if a bar is missing in one
        system.

        Parameters
        ----------
        bars : iterable of DifferentialEquation
            Bar-wise differential-equation objects of one system.

        Returns
        -------
        dict
            Mapping ``((x_i, z_i), (x_j, z_j)) -> DifferentialEquation``.
        """
        idx = {}
        for diff_eq in bars:
            idx[self._key_bar(diff_eq.bar)] = diff_eq
        return idx

    @staticmethod
    def _key_bar(
            bar: Bar,
            ndp: int = 10,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Coordinate key for bar matching between systems.

        The key is formed from the rounded coordinates of the start and end
        node of the bar. This enables robust identification even if bars are
        missing in one of the systems.

        Parameters
        ----------
        bar : Bar
            The bar for which the key is computed.

        Returns
        -------
        tuple
            ``((x1, z1), (x2, z2))`` with coordinates rounded to 10 digits.
        """
        p1 = (round(bar.node_i.x, ndp), round(bar.node_i.z, ndp))
        p2 = (round(bar.node_j.x, ndp), round(bar.node_j.z, ndp))
        return p1, p2

    @staticmethod
    def _key_node(
            node,
            ndp: int = 10,
    ) -> Tuple[float, float]:
        """
        Coordinate key for node matching between systems.

        Nodes are matched by their rounded coordinates so that small numeric
        differences from preprocessing do not prevent identification.

        Parameters
        ----------
        node : Node
            Node-object providing ``x`` and ``z``.

        Returns
        -------
        tuple of float
            ``(round(x, 10), round(z, 10))`` used as dictionary key.
        """
        return round(node.x, ndp), round(node.z, ndp)

    @staticmethod
    def _compute_moment_term(
            diff_eq_i_i: DifferentialEquation,
            diff_eq_j_i: DifferentialEquation
    ) -> float:
        """
        Bending moment contribution to the work equation for one bar pair.

        The term equals
        :math:`\\int_0^L M_1(x) M_2(x) \\frac{\\mathrm{d}x}{E I}`
        where the moments are represented by cubic polynomials.

        Parameters
        ----------
        diff_eq_i_i : DifferentialEquation
            Bending field of system ``i``.
        diff_eq_j_i : DifferentialEquation
            Bending field of system ``j``.

        Returns
        -------
        float
            Scalar bending work contribution of this bar pair.
        """
        length = diff_eq_i_i.bar.length

        z_coef_i = diff_eq_i_i.z_coef[0:4, 2]
        m1_i, m2_i, m3_i, m4_i = (
            z_coef_i[3],
            z_coef_i[2],
            z_coef_i[1],
            z_coef_i[0],
        )

        z_coef_j = diff_eq_j_i.z_coef[0:4, 2]
        m1_j, m2_j, m3_j, m4_j = (
            z_coef_j[3],
            z_coef_j[2],
            z_coef_j[1],
            z_coef_j[0],
        )

        return (1.0 / diff_eq_i_i.bar.EI) * (
                (m1_i * m1_j / 7.0) * length**7
                + (m1_i * m2_j + m2_i * m1_j) / 6.0 * length**6
                + (m1_i * m3_j + m2_i * m2_j + m3_i * m1_j) / 5.0
                * length**5
                + (m1_i * m4_j + m2_i * m3_j + m3_i * m2_j
                   + m4_i * m1_j)
                / 4.0 * length**4
                + (m2_i * m4_j + m3_i * m3_j + m4_i * m2_j)
                / 3.0 * length**3
                + (m3_i * m4_j + m4_i * m3_j) / 2.0 * length**2
                + (m4_i * m4_j) * length
        )

    @staticmethod
    def _compute_shear_term(
            diff_eq_i_i: DifferentialEquation,
            diff_eq_j_i: DifferentialEquation
    ) -> float:
        """
        Shear force contribution to the work equation for one bar pair.

        The term equals
        :math:`\\int_0^L V_1(x) V_2(x) \\frac{\\kappa}{G A}\\,\\mathrm{d}x`
        with quadratic polynomials for the shear force and a constant shear
        correction factor ``kappa``.

        Parameters
        ----------
        diff_eq_i_i : DifferentialEquation
            Shear field of system ``i``.
        diff_eq_j_i : DifferentialEquation
            Shear field of system ``j``.

        Returns
        -------
        float
            Scalar shear work contribution of this bar pair.
        """
        length = diff_eq_i_i.bar.length
        kappa = diff_eq_i_i.bar.cross_section.shear_cor
        ga = (
                diff_eq_i_i.bar.material.shear_mod
                * diff_eq_i_i.bar.cross_section.area
        )

        v_coef_i = diff_eq_i_i.z_coef[0:3, 1]
        v1_i, v2_i, v3_i = (v_coef_i[2], v_coef_i[1], v_coef_i[0])

        v_coef_j = diff_eq_j_i.z_coef[0:3, 1]
        v1_j, v2_j, v3_j = (v_coef_j[2], v_coef_j[1], v_coef_j[0],)

        return (kappa / ga) * (
                (v1_i * v1_j) / 5.0 * length**5
                + (v1_i * v2_j + v2_i * v1_j) / 4.0 * length**4
                + (v1_i * v3_j + v2_i * v2_j + v3_i * v1_j) / 3.0
                * length**3
                + (v2_i * v3_j + v3_i * v2_j) / 2.0 * length**2
                + (v3_i * v3_j) * length
        )

    @staticmethod
    def _compute_normal_term(
            diff_eq_i_i: DifferentialEquation,
            diff_eq_j_i: DifferentialEquation
    ) -> float:
        """
        Normal force contribution to the work equation for one bar pair.

        The term equals
        :math:`\\int_0^L N_1(x) N_2(x) \\frac{\\mathrm{d}x}{E A}` with
        quadratic polynomials for the normal force.

        Parameters
        ----------
        diff_eq_i_i : DifferentialEquation
            Axial field of system ``i``.
        diff_eq_j_i : DifferentialEquation
            Axial field of system ``j``.

        Returns
        -------
        float
            Scalar normal work contribution of this bar pair.
        """
        length = diff_eq_i_i.bar.length

        x_coef_i = diff_eq_i_i.x_coef[0:3, 1]
        n1_i, n2_i, n3_i = (x_coef_i[2], x_coef_i[1], x_coef_i[0])

        x_coef_j = diff_eq_j_i.x_coef[0:3, 1]
        n1_j, n2_j, n3_j = (x_coef_j[2], x_coef_j[1], x_coef_j[0])

        return (1.0 / diff_eq_i_i.bar.EA) * (
                (n1_i * n1_j) / 5.0 * length**5
                + (n1_i * n2_j + n2_i * n1_j) / 4.0 * length**4
                + (n1_i * n3_j + n2_i * n2_j + n3_i * n1_j) / 3.0
                * length**3
                + (n2_i * n3_j + n3_i * n2_j) / 2.0 * length**2
                + (n3_i * n3_j) * length
        )

    @staticmethod
    def _compute_temperature_term(
            diff_eq_i_i: DifferentialEquation,
            diff_eq_j_i: DifferentialEquation,
    ) -> Tuple[float, float]:
        """
        Temperature induced work contributions for one bar pair.

        The method returns two scalars. The first is the axial component from
        constant temperature change. The second is the bending component from a
        linear temperature gradient across the section height.

        Parameters
        ----------
        diff_eq_i_i : DifferentialEquation
            Temperature load state of system ``i``.
        diff_eq_j_i : DifferentialEquation
            Temperature load state of system ``j``.

        Returns
        -------
        tuple of float
            Axial temperature work and bending temperature work.
        """
        length = diff_eq_i_i.bar.length
        height = diff_eq_i_i.bar.cross_section.height
        alpha_t = diff_eq_i_i.bar.material.therm_exp_coeff

        t_s = diff_eq_i_i.bar.temp.temp_s
        d_t = diff_eq_i_i.bar.temp.temp_delta

        x_coef_j = diff_eq_j_i.x_coef[0:3, 1]
        n1_j, n2_j, n3_j = (x_coef_j[2], x_coef_j[1], x_coef_j[0])

        z_coef_j = diff_eq_j_i.z_coef[0:4, 2]
        m1_j, m2_j, m3_j, m4_j = (
            z_coef_j[3],
            z_coef_j[2],
            z_coef_j[1],
            z_coef_j[0]
        )

        axial = t_s * alpha_t * (
                n1_j / 3.0 * length**3
                + n2_j / 2.0 * length**2
                + n3_j * length
        )
        bending = (d_t * alpha_t / height) * (
                m1_j / 4.0 * length**4
                + m2_j / 3.0 * length**3
                + m3_j / 2.0 * length**2
                + m4_j * length
        )
        return axial, bending

    @staticmethod
    def _compute_elastic_support_term(
            node: Node,
            deform_i: np.ndarray,
            deform_j: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Elastic support work contributions at a node pair.

        Springs are interpreted as numeric stiffness values in the node
        attributes ``u``, ``w``, ``phi``. The term equals the product of real
        deformation, virtual deformation and the spring stiffness per degree of
        freedom.

        Parameters
        ----------
        node
            Node-object to describe a Node
        deform_i : array_like, shape (3,)
            Nodal deformation vector of the system  ``i``.
        deform_j : array_like, shape (3,)
            Nodal deformation vector of the system  ``i``.

        Returns
        -------
        tuple of float
            First value is the translational elastic work in the u and w
            directions, second value is the rotational elastic work around
            phi.
        """

        elastic_support = 0.0
        elastic_fixed_support = 0.0

        if isinstance(node.u, (int, float)):
            elastic_support += float(
                deform_i[0] * deform_j[0] * node.u
            )

        if isinstance(node.w, (int, float)):
            elastic_support += float(
                deform_i[1] * deform_j[1] * node.w
            )

        if isinstance(node.phi, (int, float)):
            elastic_fixed_support += float(
                deform_i[2] * deform_j[2] * node.phi
            )

        return elastic_support, elastic_fixed_support

    @staticmethod
    def _compute_displacement_term(
            node: Node,
            deform_j: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Imposed displacement work contributions at a node pair.

        Translations in x and z as well as the rotation phi are taken from the
        imposed displacement set of the real node and combined with the
        virtual reaction components at the same node.

        Parameters
        ----------
        node
            Node-object to describe a Node
        deform_j : array_like, shape (3,)
            Nodal deformation vector of the system  ``j``.

        Returns
        -------
        tuple of float
            First value is the contribution from lateral translations u and w,
            second value is the contribution from rotation phi.
        """
        translational_displacement = 0.0
        rotational_displacement = 0.0

        for disp in node.displacements:
            translational_displacement += float(
                -disp.x * deform_j[0]
                - disp.z * deform_j[1]
            )
            rotational_displacement += float(
                -disp.phi * deform_j[2]
            )
        return translational_displacement, rotational_displacement

    @staticmethod
    def _sum_matrix(m: np.ndarray):
        """
        Column-wise sum of a work matrix.

        Parameters
        ----------
        m : np.ndarray
            Matrix produced by bar or node work assembly.

        Returns
        -------
        np.ndarray
            1D array of the column sums.
        """
        row_sums = np.sum(m, axis=1, keepdims=True)
        matrix_with_row = np.hstack([m, row_sums])

        col_sums = np.sum(matrix_with_row, axis=0, keepdims=True)
        matrix_final = np.vstack([matrix_with_row, col_sums])
        return matrix_final
