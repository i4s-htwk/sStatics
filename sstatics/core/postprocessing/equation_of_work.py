from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

import numpy as np
from tabulate import tabulate

from sstatics.core.postprocessing.results import (
    SystemResult, BarResult, NodeResult)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EquationOfWork:
    """
    Assemble and evaluate the terms of the work equation for two systems.

    The class combines bar based and node based contributions between a real
    system and a virtual system. It provides componentwise term evaluators for
    bending, shear, normal force and temperature effects as well as elastic
    and imposed displacements at nodes. Results can be aggregated into
    matrices for bars and nodes and logged in tabular form.

    Parameters
    ----------
    result_system_1 : SystemResult
        Postprocessing result container of the real system. Must provide
        aligned iterables ``bars`` and ``nodes`` with the required fields used
        below.
    result_system_2 : SystemResult
        Postprocessing result container of the virtual system. The bars and
        nodes are matched to those of ``result_system_1`` by their end node
        coordinates.

    Notes
    -----
    The formulas assume polynomial representations of internal force per bar
    with coefficient arrays accessible via ``x_coef`` and ``z_coef``. Material
    and section properties are read through the nested attributes of each bar
    result.
    """

    result_system_1: SystemResult
    result_system_2: SystemResult

    @staticmethod
    def compute_moment_term(br_real: BarResult, br_virt: BarResult) -> float:
        """
        Bending moment contribution to the work equation for one bar pair.

        The term equals
        :math:`\\int_0^L M_1(x) M_2(x) \\frac{\\mathrm{d}x}{E I}`
        where the moments are represented by cubic polynomials.

        Parameters
        ----------
        br_real
            Bar result of the real system with attributes ``bar``, ``z_coef``
            and section properties.
        br_virt
            Matching bar result of the virtual system.

        Returns
        -------
        float
            Scalar bending work contribution of this bar pair.
        """
        length = br_real.bar.length
        ei = br_real.bar.EI

        z_coef_real = br_real.z_coef[0:4, 2]
        m1_real, m2_real, m3_real, m4_real = (
            z_coef_real[3],
            z_coef_real[2],
            z_coef_real[1],
            z_coef_real[0],
        )

        z_coef_virt = br_virt.z_coef[0:4, 2]
        m1_virt, m2_virt, m3_virt, m4_virt = (
            z_coef_virt[3],
            z_coef_virt[2],
            z_coef_virt[1],
            z_coef_virt[0],
        )

        return (1.0 / ei) * (
            (m1_real * m1_virt / 7.0) * length**7
            + (m1_real * m2_virt + m2_real * m1_virt) / 6.0 * length**6
            + (m1_real * m3_virt + m2_real * m2_virt + m3_real * m1_virt) / 5.0
            * length**5
            + (m1_real * m4_virt + m2_real * m3_virt + m3_real * m2_virt
                + m4_real * m1_virt)
            / 4.0 * length**4
            + (m2_real * m4_virt + m3_real * m3_virt + m4_real * m2_virt)
            / 3.0 * length**3
            + (m3_real * m4_virt + m4_real * m3_virt) / 2.0 * length**2
            + (m4_real * m4_virt) * length
        )

    @staticmethod
    def compute_shear_term(br_real: BarResult, br_virt: BarResult) -> float:
        """
        Shear force contribution to the work equation for one bar pair.

        The term equals
        :math:`\\int_0^L V_1(x) V_2(x) \\frac{\\kappa}{G A}\\,\\mathrm{d}x`
        with quadratic polynomials for the shear force and a constant shear
        correction factor ``kappa``.

        Parameters
        ----------
        br_real
            Bar result of the real system with attributes ``bar``, ``z_coef``
            and section properties.
        br_virt
            Matching bar result of the virtual system.

        Returns
        -------
        float
            Scalar shear work contribution of this bar pair.
        """
        length = br_real.bar.length
        kappa = br_real.bar.cross_section.shear_cor
        ga = (
            br_real.bar.material.shear_mod
            * br_real.bar.cross_section.area
        )

        v_coef_real = br_real.z_coef[0:3, 1]
        v1_real, v2_real, v3_real = (
            v_coef_real[2],
            v_coef_real[1],
            v_coef_real[0],
        )

        v_coef_virt = br_virt.z_coef[0:3, 1]
        v1_virt, v2_virt, v3_virt = (
            v_coef_virt[2],
            v_coef_virt[1],
            v_coef_virt[0],
        )

        return (kappa / ga) * (
            (v1_real * v1_virt) / 5.0 * length**5
            + (v1_real * v2_virt + v2_real * v1_virt) / 4.0 * length**4
            + (v1_real * v3_virt + v2_real * v2_virt + v3_real * v1_virt) / 3.0
            * length**3
            + (v2_real * v3_virt + v3_real * v2_virt) / 2.0 * length**2
            + (v3_real * v3_virt) * length
        )

    @staticmethod
    def compute_normal_term(br_real: BarResult, br_virt: BarResult) -> float:
        """
        Normal force contribution to the work equation for one bar pair.

        The term equals
        :math:`\\int_0^L N_1(x) N_2(x) \\frac{\\mathrm{d}x}{E A}` with
        quadratic polynomials for the normal force.

        Parameters
        ----------
        br_real
            Bar result of the real system with attributes ``bar``, ``x_coef``
            and section properties.
        br_virt
            Matching bar result of the virtual system.

        Returns
        -------
        float
            Scalar normal work contribution of this bar pair.
        """
        length = br_real.bar.length
        ea = br_real.bar.EA

        x_coef_real = br_real.x_coef[0:3, 1]
        n1_real, n2_real, n3_real = (
            x_coef_real[2],
            x_coef_real[1],
            x_coef_real[0],
        )

        x_coef_virt = br_virt.x_coef[0:3, 1]
        n1_virt, n2_virt, n3_virt = (
            x_coef_virt[2],
            x_coef_virt[1],
            x_coef_virt[0],
        )

        return (1.0 / ea) * (
            (n1_real * n1_virt) / 5.0 * length**5
            + (n1_real * n2_virt + n2_real * n1_virt) / 4.0 * length**4
            + (n1_real * n3_virt + n2_real * n2_virt + n3_real * n1_virt) / 3.0
            * length**3
            + (n2_real * n3_virt + n3_real * n2_virt) / 2.0 * length**2
            + (n3_real * n3_virt) * length
        )

    @staticmethod
    def compute_temperature_term(
        br_real: BarResult,
        br_virt: BarResult,
    ) -> Tuple[float, float]:
        """
        Temperature induced work contributions for one bar pair.

        The method returns two scalars. The first is the axial component from
        constant temperature change. The second is the bending component from a
        linear temperature gradient across the section height.

        Parameters
        ----------
        br_real
            Bar result of the real system with attributes ``bar`` and ``temp``
            including ``temp_s`` and ``temp_delta``.
        br_virt
            Matching bar result of the virtual system with coefficient arrays
            ``x_coef`` and ``z_coef``.

        Returns
        -------
        tuple of float
            Axial temperature work and bending temperature work.
        """
        length = br_real.bar.length
        height = br_real.bar.cross_section.height
        alpha_t = br_real.bar.material.therm_exp_coeff

        t_s = br_real.bar.temp.temp_s
        d_t = br_real.bar.temp.temp_delta

        x_coef_virt = br_virt.x_coef[0:3, 1]
        n1_virt, n2_virt, n3_virt = (
            x_coef_virt[2],
            x_coef_virt[1],
            x_coef_virt[0],
        )

        z_coef_virt = br_virt.z_coef[0:4, 2]
        m1_virt, m2_virt, m3_virt, m4_virt = (
            z_coef_virt[3],
            z_coef_virt[2],
            z_coef_virt[1],
            z_coef_virt[0],
        )

        axial = t_s * alpha_t * (
            n1_virt / 3.0 * length**3
            + n2_virt / 2.0 * length**2
            + n3_virt * length
        )
        bending = (d_t * alpha_t / height) * (
            m1_virt / 4.0 * length**4
            + m2_virt / 3.0 * length**3
            + m3_virt / 2.0 * length**2
            + m4_virt * length
        )
        return axial, bending

    @staticmethod
    def compute_elastic_support_term(
        nr_real: NodeResult,
        nr_virt: NodeResult,
    ) -> Tuple[float, float]:
        """
        Elastic support work contributions at a node pair.

        Springs are interpreted as numeric stiffness values in the node
        attributes ``u``, ``w``, ``phi``. The term equals the product of real
        deformation, virtual deformation and the spring stiffness per degree of
        freedom.

        Parameters
        ----------
        nr_real
            Node result of the real system. Uses ``node.u``, ``node.w``,
            ``node.phi`` and the corresponding entries of ``deform``.
        nr_virt
            Matching node result of the virtual system.

        Returns
        -------
        tuple of float
            First value is the translational elastic work in the u and w
            directions, second value is the rotational elastic work around
            phi.
        """

        elastic_support = 0.0
        elastic_fixed_support = 0.0

        if isinstance(nr_real.node.u, (int, float)):
            elastic_support += float(
                nr_real.deform[0] * nr_virt.deform[0] * nr_real.node.u
            )

        if isinstance(nr_real.node.w, (int, float)):
            elastic_support += float(
                nr_real.deform[1] * nr_virt.deform[1] * nr_real.node.w
            )

        if isinstance(nr_real.node.phi, (int, float)):
            elastic_fixed_support += float(
                nr_real.deform[2] * nr_virt.deform[2] * nr_real.node.phi
            )

        return elastic_support, elastic_fixed_support

    @staticmethod
    def compute_displacement_term(
        nr_real: NodeResult,
        nr_virt: NodeResult,
    ) -> Tuple[float, float]:
        """
        Imposed displacement work contributions at a node pair.

        Translations in x and z as well as the rotation phi are taken from the
        imposed displacement set of the real node and combined with the
        virtual reaction components at the same node.

        Parameters
        ----------
        nr_real
            Node result of the real system. Uses iterable
            ``node.displacements`` with fields ``x``, ``z``, ``phi``.
        nr_virt
            Matching node result of the virtual system. Uses ``node_support``
            of length three as virtual reactions in u, w and phi.

        Returns
        -------
        tuple of float
            First value is the contribution from lateral translations u and w,
            second value is the contribution from rotation phi.
        """
        translational_displacement = 0.0
        rotational_displacement = 0.0

        for disp in nr_real.node.displacements:
            translational_displacement += float(
                -disp.x * nr_virt.node_support[0]
                - disp.z * nr_virt.node_support[1]
            )
            rotational_displacement += float(
                -disp.phi * nr_virt.node_support[2]
            )
        return translational_displacement, rotational_displacement

    @staticmethod
    def _key_bar(
        br,
        ndp: int = 10,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Create a coordinate key for a bar result.

        Parameters
        ----------
        br
            Bar result object with nested ``bar.node_i`` and ``bar.node_j``
            carrying coordinates ``x`` and ``z``.
        ndp : int, optional
            Number of decimal places used for rounding, default is ten.

        Returns
        -------
        tuple
            Pair of coordinate tuples for the start and end node.
        """
        p1 = (round(br.bar.node_i.x, ndp), round(br.bar.node_i.z, ndp))
        p2 = (round(br.bar.node_j.x, ndp), round(br.bar.node_j.z, ndp))
        return p1, p2

    def _index_bars_by_key(
        self,
        bars,
    ) -> Dict[Tuple[Tuple[float, float], Tuple[float, float]], List]:
        """
        Build an index of bar results keyed by their rounded coordinates.

        Parameters
        ----------
        bars
            Iterable of bar results.

        Returns
        -------
        dict
            Mapping from the coordinate key to a list of bar results.
        """
        idx: Dict[Tuple[Tuple[float, float], Tuple[float, float]], List] = {}
        for br in bars:
            idx.setdefault(self._key_bar(br), []).append(br)
        return idx

    @staticmethod
    def _key_node(
        nr,
        ndp: int = 10,
    ) -> Tuple[float, float]:
        """
        Create a coordinate key for a node result.

        Parameters
        ----------
        nr
            Node result object with ``node.x`` and ``node.z``.
        ndp : int, optional
            Number of decimal places used for rounding, default is ten.

        Returns
        -------
        tuple of float
            Rounded coordinate pair.
        """
        return round(nr.node.x, ndp), round(nr.node.z, ndp)

    def _index_nodes_by_key(
        self,
        nodes,
    ) -> Dict[Tuple[float, float], List]:
        """
        Build an index of node results keyed by their rounded coordinates.

        Parameters
        ----------
        nodes
            Iterable of node results.

        Returns
        -------
        dict
            Mapping from the coordinate key to a list of node results.
        """
        idx: Dict[Tuple[float, float], List] = {}
        for nr in nodes:
            idx.setdefault(self._key_node(nr), []).append(nr)
        return idx

    def _create_work_matrix_bars(self) -> np.ndarray:
        """
        Assemble bar wise work contributions into a matrix.

        Columns are ordered as Moment, Normal, Shear, Temperature constant,
        Temperature delta.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_bars, 5)`` with zeros where no matching
            virtual bar exists.
        """
        work_matrix_bars = np.zeros((len(self.result_system_1.bars), 5))
        virt_idx = self._index_bars_by_key(self.result_system_2.bars)

        for i, br_real in enumerate(self.result_system_1.bars):
            key = self._key_bar(br_real)
            br_virt = virt_idx.get(key, [None])[0]
            if br_virt is None:
                continue

            if "moment" in br_real.bar.deformations:
                work_matrix_bars[i, 0] += self.compute_moment_term(
                    br_real,
                    br_virt,
                )
            if "normal" in br_real.bar.deformations:
                work_matrix_bars[i,  1] += self.compute_normal_term(
                    br_real,
                    br_virt,
                )
            if "shear" in br_real.bar.deformations:
                work_matrix_bars[i, 2] += self.compute_shear_term(
                    br_real,
                    br_virt,
                )

            work_matrix_bars[i, 3:5] += self.compute_temperature_term(
                br_real,
                br_virt,
            )

        return work_matrix_bars

    def _create_work_matrix_nodes(self) -> np.ndarray:
        """
        Assemble node wise work contributions into a matrix.

        Columns are ordered as Elastic support translational, Elastic support
        rotational, Imposed displacements u and w, Imposed rotation phi.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_nodes, 4)`` with zeros where no matching
            virtual node exists.
        """
        work_matrix_nodes = np.zeros((len(self.result_system_1.nodes), 4))
        virt_idx = self._index_nodes_by_key(self.result_system_2.nodes)

        for i, nr_real in enumerate(self.result_system_1.nodes):
            key = self._key_node(nr_real)
            nr_virt = virt_idx.get(key, [None])[0]
            if nr_virt is None:
                continue

            work_matrix_nodes[i, 0:2] += self.compute_elastic_support_term(
                nr_real,
                nr_virt,
            )

            if nr_real.node.displacements:
                work_matrix_nodes[i, 2:4] += self.compute_displacement_term(
                    nr_real,
                    nr_virt,
                )
        return work_matrix_nodes

    @staticmethod
    def _sum_matrix(matrix):
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        matrix_with_row = np.hstack([matrix, row_sums])

        col_sums = np.sum(matrix_with_row, axis=0, keepdims=True)
        matrix_final = np.vstack([matrix_with_row, col_sums])
        return matrix_final

    def log_work_contributions(
            self,
            delta: str = "m",
            decimals: int = 6
    ) -> None:
        """
        Log tabulated work contributions per bar and per node.

        Sums are appended as last column and last row to each table. Output is
        sent to the module logger.

        Parameters
        ----------
        delta : str, optional
            Identifier of the considered generalized displacement, appended to
            the table titles. Default is ``"m"``.
        decimals : int, optional
            Number of decimal places used for floating point formatting in
            tables. Default is ``6``.
        """
        bars = self._create_work_matrix_bars()
        nodes = self._create_work_matrix_nodes()
        #
        # from sstatics.core.logger_mixin import table_matrix
        # print(table_matrix(matrix=self._sum_matrix(bars),
        #                    column_names=[
        #                        "Bar",
        #                        "Moment",
        #                        "Normal",
        #                        "Shear",
        #                        "Temperature Constant",
        #                        "Temperature Delta",
        #                        "Sum",
        #                    ]))
        #
        # print(table_matrix(matrix=self._sum_matrix(nodes),
        #                    column_names=[
        #                        "Node",
        #                        "Elastic Support F",
        #                        "Elastic Support M",
        #                        "Displacements u and w",
        #                        "Displacements phi",
        #                        "Sum",
        #                    ]))

        # Bars table with row and column sums
        bar_numbers = np.arange(1, bars.shape[0] + 1).astype(str)
        bar_numbers = np.append(bar_numbers, "Sum").reshape(-1, 1)

        row_sums_b = np.sum(bars, axis=1, keepdims=True)
        bars_with_row_sums = np.column_stack((bars, row_sums_b))
        column_sums_b = np.sum(bars_with_row_sums, axis=0, keepdims=True)
        bars_with_sums = np.vstack((bars_with_row_sums, column_sums_b))
        bars_with_sums = np.hstack((bar_numbers, bars_with_sums))

        table_bars = tabulate(
            bars_with_sums,
            headers=[
                "Bar",
                "Moment",
                "Normal",
                "Shear",
                "Temperature Constant",
                "Temperature Delta",
                "Sum",
            ],
            tablefmt="grid",
            floatfmt=f".{decimals}f",
        )

        # Nodes table with row and column sums
        node_numbers = np.arange(1, nodes.shape[0] + 1).astype(str)
        node_numbers = np.append(node_numbers, "Sum").reshape(-1, 1)

        row_sums_n = np.sum(nodes, axis=1, keepdims=True)
        nodes_with_row_sums = np.column_stack((nodes, row_sums_n))
        column_sums_n = np.sum(nodes_with_row_sums, axis=0, keepdims=True)
        nodes_with_sums = np.vstack((nodes_with_row_sums, column_sums_n))
        nodes_with_sums = np.hstack((node_numbers, nodes_with_sums))

        table_nodes = tabulate(
            nodes_with_sums,
            headers=[
                "Node",
                "Elastic Support F",
                "Elastic Support M",
                "Displacements u and w",
                "Displacements phi",
                "Sum",
            ],
            tablefmt="grid",
            floatfmt=f".{decimals}f",
        )

        logger.info(
            "Work contributions of delta_%s for each bar:\n%s",
            delta,
            table_bars,
        )
        logger.info(
            "Work contributions of delta_%s for each node:\n%s",
            delta,
            table_nodes,
        )

    def delta_s1_s2(self) -> float:
        """
        Sum of all work equation terms over bars and nodes.

        Returns
        -------
        float
            Total work interaction between system one and system two.
        """
        return float(
            np.sum(self._create_work_matrix_bars())
            + np.sum(self._create_work_matrix_nodes())
        )
