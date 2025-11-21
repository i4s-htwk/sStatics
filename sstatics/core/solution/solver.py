
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core.logger_mixin import (LoggerMixin, table_bar, table_node,
                                        table_node_bar_index)
from sstatics.core.preprocessing.system import System
from sstatics.core.utils import transformation_matrix


@dataclass(eq=False)
class Solver(LoggerMixin):
    """Executes first-order static analysis for the provided system.

    The analysis is based on the deformation method, assuming linear-elastic
    material behavior and small displacements. Suitable for systems where
    second-order effects can be neglected. The

     Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using first-order theory.
    """

    system: System
    debug: bool = False

    # INITIALIZATION -------------------------------------------------
    def __post_init__(self):
        self.logger.debug("Starting solver initialization.")
        self.logger.debug("Extracting mesh and nodes from system object.")

        self.dof = 3
        self.mesh = self.system.mesh
        self.nodes = self.system.nodes()

        self.logger.debug(
            f"Mesh successfully initialized with {len(self.nodes)} nodes."
        )

        self.logger.info("Checking system solvability...")

        if self.solvable:
            self.logger.info(
                "System is solvable. Proceeding to compute deformations.")
            mapping = table_node_bar_index(self.mesh, self.nodes)
            _ = [self.node_deform, self.system_node_deform,
                 self.node_support_forces, self.system_support_forces,
                 self.internal_forces,
                 self.bar_deform, self.bar_deform_hinge,
                 self.bar_deform_displacements, self.bar_deform_list,
                 self.system_deform_list]

            # Define all tables in a list of (description, data, headers,
            # function)
            tables = [
                ("Node deformation",
                 [self.node_deform, self.system_node_deform],
                 [['u', 'w', 'φ'], ['u\u0303', 'w\u0303', 'φ\u0303']],
                 table_node),
                ("Support forces",
                 [self.node_support_forces, self.system_support_forces],
                 [['Px', 'Pz', 'Pm'], ['Px\u0303', 'Pz\u0303', 'Pm\u0303']],
                 table_node),
                ("Bar Deformation",
                 [self.bar_deform, self.bar_deform_hinge,
                  self.bar_deform_displacements, self.bar_deform_list],
                 [["u' wgv", "w' wgv", "φ' wgv"],
                  ["u' hinge", "w' hinge", "φ' hinge"],
                  ["u' displacements", "w' displacements", "φ' displacements"],
                  ["u'", "w'", "φ'"]],
                 lambda data, header: table_bar(data, mapping, header)),
                ("Bar deformation",
                 [self.bar_deform_list, self.system_deform_list],
                 [["u'", "w'", "φ'"], ['u\u0303', 'w\u0303', 'φ\u0303']],
                 lambda data, header: table_bar(data, mapping, header)),
                ("Internal forces",
                 [self.internal_forces],
                 [["fx (N)", "fz (V_z)", "fm (M_y)"]],
                 lambda data, header: table_bar(data, mapping, header))
            ]

            # Log all tables
            for desc, data, header, func in tables:
                self.logger.debug(f"{desc}: \n{func(data, header)}")
        else:
            self.logger.error(
                "System is not solvable. Stopping solver initialization.")

    # ASSEMBLY -------------------------------------------------------
    def _get_zero_matrix(self):
        """Creates a zero matrix based on the number of nodes in the system.

        The matrix size is determined by the number of nodes, each having
        three degrees of freedom (as defined by :py:attr:`dof` = 3).

        Returns
        -------
        :any:`numpy.array`
            A square zero matrix of size (number of nodes × DOF).

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> cross_section = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        >>> material = Material(2.1e8, 0.1, 0.1, 0.1)
        >>> node1 = Node(0, 0, u='fixed', w='fixed')
        >>> node2 = Node(3, 0, w='fixed')
        >>> node3 = Node(6, 0, w='fixed')
        >>> bar1 = Bar(node1, node2, cross_section, material)
        >>> bar2 = Bar(node2, node3, cross_section, material)
        >>> system = System([bar1, bar2])
        >>> Solver(system)._get_zero_matrix()
        array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        >>> system = System([bar1])
        >>> Solver(system)._get_zero_matrix()
        array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])
        """
        x = len(self.nodes) * self.dof
        self.logger.debug(f"Creating zero matrix of size {x}x{x}")
        return np.zeros((x, x))

    def _get_zero_vec(self):
        """Creates a zero vector based on the number of nodes in the system.

        Each node is assumed to have three degrees of freedom (DOF = 3), and
        the vector represents an initial state (filled with zeros) used in the
        structural analysis.

        Returns
        -------
        :any:`numpy.array`
            A zero matrix of size (number of nodes × DOF).

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> cross_section = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        >>> material = Material(2.1e8, 0.1, 0.1, 0.1)
        >>> node1 = Node(0, 0, u='fixed', w='fixed')
        >>> node2 = Node(3, 0, w='fixed')
        >>> node3 = Node(6, 0, w='fixed')
        >>> bar1 = Bar(node1, node2, cross_section, material)
        >>> bar2 = Bar(node2, node3, cross_section, material)
        >>> system = System([bar1, bar2])
        >>> Solver(system)._get_zero_vec()
        array([[0], [0], [0], [0], [0], [0], [0], [0], [0]])
        >>> system = System([bar1])
        >>> Solver(system)._get_zero_vec()
        array([[0], [0], [0], [0], [0], [0]])
        """
        x = len(self.nodes) * self.dof
        self.logger.debug(f"Creating zero vector of size {x}x{x}")
        return np.zeros((x, 1))

    @cached_property
    def stiffness_matrix(self):
        r"""Generates the global stiffness matrix of the system, which is
        formed from the element stiffness matrices of the individual basic
        elements.

        To represent the stiffness relationships of the entire system, the
        global stiffness matrix is assembled by considering how the bars are
        connected at the nodes. This is needed to fulfill the equilibrium
        conditions on each node.

        Returns
        -------
        :any:`numpy.array`
            A matrix whose dimensions are determined by the number of nodes.
            For a system with 3 nodes, due to the degrees of freedom (3), the
            resulting matrix will be a 9x9 matrix.

        Notes
        -----
            The global stiffness matrix is constructed by assembling the
            element stiffness matrices of the individual bars. Initially, the
            matrix is filled with zeros, and then the element stiffness
            matrices are placed in the corresponding locations. If multiple
            bars intersect at a node, the element stiffness matrices are
            superimposed at that node.
        """
        self.logger.info("Starting assembly of global stiffness matrix")
        k_system = self._get_zero_matrix()
        dof = self.dof
        total_bars = len(self.mesh)
        self.logger.debug(f"Processing {total_bars} elements in the mesh")

        for bar_idx, bar in enumerate(self.mesh):
            self.logger.debug(f"Processing element {bar_idx + 1}/{total_bars}")
            i, j = (self.nodes.index(bar.node_i) * dof,
                    self.nodes.index(bar.node_j) * dof)
            k = bar.stiffness_matrix()
            self.logger.debug(f"Element Stiffness matrix: \n {repr(k)}")

            k_system[i:i + dof, i:i + dof] += k[:dof, :dof]
            k_system[i:i + dof, j:j + dof] += k[:dof, dof:2 * dof]
            k_system[j:j + dof, i:i + dof] += k[dof:2 * dof, :dof]
            k_system[j:j + dof, j:j + dof] += k[dof:2 * dof, dof:2 * dof]

        self.logger.info(
            f"Stiffness matrix assembly completed. Matrix shape: "
            f"{k_system.shape}")
        self.logger.debug(f"Stiffness matrix: \n {repr(k_system)}")
        return k_system

    @cached_property
    def elastic_matrix(self):
        """Generates a matrix that accounts for the elastic support at the
        nodes.

        The elastic matrix represents the stiffness of the elastic supports
        at each node, which are placed along the diagonal. If the node is
        elastically supported, the matrix will contain the corresponding spring
        or rotational spring stiffness values.

        Returns
        -------
        :any:`numpy.array`
            A matrix whose dimensions are determined by the number of nodes
            and the degrees of freedom. For a system with 3 nodes and 3 degrees
            of freedom, the resulting matrix will be 9x9. The diagonal of the
            matrix contains the spring and rotational spring stiffness values
            for the supports at each node, if the node is elastically
            supported.

        Notes
        -----
            The elastic matrix has the same dimensions as the
            :py:attr:`stiffness_matrix`. The diagonal elements of the matrix
            contain the stiffness values of the support components for each
            node.
        """
        self.logger.info("Starting assembly of elastic support matrix")
        elastic = self._get_zero_matrix()
        dof = self.dof
        total_bars = len(self.mesh)
        self.logger.debug(
            f"Processing elastic supports for {total_bars} elements")

        for bar_idx, bar in enumerate(self.mesh):
            self.logger.debug(f"Processing element {bar_idx + 1}/{total_bars}")
            i, j = (self.nodes.index(bar.node_i) * dof,
                    self.nodes.index(bar.node_j) * dof)

            el_bar = np.block([
                [bar.node_i.elastic_support, np.zeros((dof, dof))],
                [np.zeros((dof, dof)), bar.node_j.elastic_support]
            ])
            self.logger.debug(f"Element elastic support matrix: \n"
                              f"{repr(el_bar)}")

            elastic[i:i + dof, i:i + dof] = el_bar[:dof, :dof]
            elastic[i:i + dof, j:j + dof] = el_bar[:dof, dof:2 * dof]
            elastic[j:j + dof, i:i + dof] = el_bar[dof:2 * dof, :dof]
            elastic[j:j + dof, j:j + dof] = el_bar[dof:2 * dof, dof:2 * dof]

        self.logger.info(
            f"Elastic matrix assembly completed. Matrix shape: "
            f"{elastic.shape}")
        self.logger.debug(
            f"Diagonal of the elastic matrix: \n {np.diag(elastic)}")
        return elastic

    @cached_property
    def system_matrix(self):
        """Adds the global stiffness matrix and the elastic matrix to form the
        system stiffness matrix.

        Returns
        -------
        :any:`numpy.array`
            Sum of :py:attr:`stiffness_matrix` and :py:attr:`elastic_matrix`,
            resulting in the system stiffness matrix.
        """
        self.logger.info(
            "Creating system matrix by combining stiffness and elastic "
            "matrices")
        system_matrix = self.stiffness_matrix + self.elastic_matrix
        self.logger.info(
            f"System matrix created. Matrix shape: {system_matrix.shape}")
        self.logger.debug(f"System matrix: \n {repr(system_matrix)}")
        return system_matrix

    @cached_property
    def f0(self):
        """Calculates the total force vector of the bar internal forces from
        the applied loads.

        Returns
        -------
        :any:`numpy.array`
            A vector with dimensions (number of nodes * dof (=3) x 1),
            which contains the internal forces of the bars resulting from the
            applied loads.

        Notes
        -----
            The vector is constructed by adding the force vectors of the
            individual bars in the system. These vectors are placed in the
            correct rows based on the node indices. The function iterates over
            all the bars in the system to assemble the total force vector.
        """
        self.logger.info("Starting assembly of internal force vector (f0)")
        f0_system = self._get_zero_vec()
        dof = self.dof
        total_bars = len(self.mesh)
        self.logger.debug(f"Processing {total_bars} elements to calculate f0")

        for bar_idx, bar in enumerate(self.mesh):
            self.logger.debug(f"Processing element {bar_idx + 1}/{total_bars}")
            i, j = (self.nodes.index(bar.node_i) * dof,
                    self.nodes.index(bar.node_j) * dof)
            f0 = bar.f0()
            self.logger.debug(f"Element Internal force vector: \n"
                              f"{repr(f0)}")

            f0_system[i:i + dof, :] += f0[:dof, :]
            f0_system[j:j + dof, :] += f0[dof:2 * dof, :]

        self.logger.info(
            f"Internal force vector (f0) assembly completed. Vector shape: "
            f"{f0_system.shape}")
        self.logger.debug(f"Internal force vector: \n {repr(f0_system)}")
        return f0_system

    @cached_property
    def p0(self):
        """Calculates the total vector of external node loads.

        The external node loads are considered by collecting them in a total
        load vector during the calculation.

        Returns
        -------
        :any:`numpy.array`
            A vector with dimensions (number of nodes * dof (=3) x 1),
            which contains the external node loads.

        Notes
        -----
            Similar to the vector :py:attr:`f0`, the external node loads are
            assembled in the vector based on the node indices.
        """
        self.logger.info("Starting assembly of external load vector (p0)")
        p0 = self._get_zero_vec()
        total_nodes = len(self.nodes)
        self.logger.debug(f"Processing {total_nodes} nodes to calculate p0")

        for node_idx, node in enumerate(self.nodes):
            self.logger.debug(f"Processing node {node_idx + 1}/{total_nodes}")
            p0[node_idx * self.dof:node_idx * self.dof + self.dof, :] = (
                node.load)

        self.logger.info(
            f"External load vector (p0) assembly completed. Vector shape: "
            f"{p0.shape}")
        self.logger.debug(f"External load vector: {p0}")
        return p0

    @cached_property
    def p(self):
        r"""Computes the global load vector of the system.

        The global load vector is obtained by subtracting the internal nodal
        force vector from the external load vector.

        Returns
        -------
        :any:`numpy.array`
            The global load vector of the system.

        Notes
        -----
            The global load vector `P` can be expressed as the product of the
            global stiffness matrix `k` and the nodal displacement vector
            :math:`\Delta`:

            .. math::
                P = k \cdot \Delta

            In displacement-based methods, the nodal displacements
            :math:`\Delta` are the unknowns to be solved for. The global load
            vector is computed from known loads, while the stiffness matrix `k`
            is also a known quantity.
        """
        self.logger.info(
            "Calculating global load vector by combining external and "
            "internal forces")
        global_p = self.p0 - self.f0
        self.logger.info(
            f"Global load vector (p) calculated. Vector shape: "
            f"{global_p.shape}")
        self.logger.debug(f"Global load vector: \n {repr(global_p)}")
        return global_p

    # APPLY BOUNDARY CONDITIONS --------------------------------------
    @cached_property
    def boundary_conditions(self):
        """Applies boundary conditions to the global system.

        This function modifies the :py:attr:`system_matrix` and the total load
        vector :py:attr:`p` to account for the boundary conditions (supports)
        of each node in the structure. It ensures that the resulting system of
        equations is solvable by eliminating degrees of freedom that are fixed.

        Returns
        -------
        :any:`tuple` of `numpy.array`
            A tuple `(k, p)` where `k` is the modified global stiffness matrix,
            and `p` is the modified global load vector after applying boundary
            conditions.

        Notes
        -----
            The initially assembled
            :py:attr:`system_matrix` is singular, as it does not yet include
            support conditions. Its determinant is zero, which makes the
            equation system unsolvable.

            To make the system solvable, it is essential to incorporate support
            conditions.
            When a deformation is restricted (i.e., if the attributes
            :py:attr:`u`, :py:attr:`w`, or :py:attr:`phi` are set to 'fixed'),
            the corresponding entry in the nodal displacement vector
            :math:`\\Delta` is set to zero.

            Instead of removing rows and columns, which would require
            restructuring the system, this algorithm zeroes out the
            corresponding rows and columns in the stiffness matrix `k` and sets
            the diagonal entry to one. This maintains the shape of the result
            vector :math:`\\Delta` . The same positions in the load vector
            :py:attr:`p` are set to zero.

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> from sstatics.core.preprocessing.loads import BarLineLoad
        >>> cross_section = CrossSection(1940e-8, 28.5e-4, 0.2, 0.1, 0.1)
        >>> material = Material(2.1e8, 0.1, 0.1, 0.1)
        >>> node1 = Node(0, 0, u='fixed', w='fixed')
        >>> node2 = Node(3, 0, w='fixed')
        >>> load = BarLineLoad(1, 1, 'z', 'bar', 'exact')
        >>> bar1 = Bar(node1, node2, cross_section, material, line_loads=load)
        >>> system = System([bar1])
        >>> Solver(system).system_matrix
        array([
        [199500, 0, 0, -199500, 0, 0],
        [0, 1810.67, -2716, 0, -1810.67, -2716],
        [0, -2716, 5432, 0, 2716, 2716],
        [-199500, 0, 0, 199500, 0, 0],
        [0, -1810.67, 2716, 0, 1810.67, 2716],
        [0, -2716, 2716, 0, 2716, 5432]])
        >>> Solver(system).p
        array([[0], [1.5], [-0.75], [0], [1.5], [0.75]])
        >>> Solver(system).boundary_conditions
        (array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 5432, 0, 0, 2716],
        [0, 0, 0, 199500, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 2716, 0, 0, 5432]]),
        array([[0], [0], [-0.75], [0], [0], [0.75]]))
        """
        self.logger.info("Applying boundary conditions to system")
        k = deepcopy(self.system_matrix)
        p = deepcopy(self.p)
        total_nodes = len(self.nodes)
        fixed_dofs = 0
        self.logger.debug(
            f"Checking boundary conditions for {total_nodes} nodes")

        for idx, node in enumerate(self.nodes):
            node_offset = idx * self.dof
            for dof_nr, attribute in enumerate(['u', 'w', 'phi']):
                if getattr(node, attribute, 'free') == 'fixed':
                    self.logger.debug(
                        f"Fixed DOF found: node {idx + 1}, DOF {dof_nr} "
                        f"({attribute})")
                    k[node_offset + dof_nr, :] = 0
                    k[:, node_offset + dof_nr] = 0
                    k[node_offset + dof_nr, node_offset + dof_nr] = 1
                    p[node_offset + dof_nr] = 0
                    fixed_dofs += 1

        self.logger.info(
            f"Boundary conditions applied. Fixed {fixed_dofs} DOFs")
        self.logger.debug(f"Modified Stiffness matrix: \n {repr(k)}")
        self.logger.debug(f"Modified Global load vector: \n {repr(p)}")
        return k, p

    # SOLVE SYSTEM ---------------------------------------------------
    @cached_property
    def solvable(self):
        """Checks whether the stiffness matrix is regular, i.e., whether the
        system of equations is solvable.

        Returns
        -------
        :any:`bool`
            :python:`False` if the stiffness matrix is singular (unsolvable),
            :python:`True` otherwise.

        Notes
        -----
        This method checks whether the system is kinematically movable by
        examining the rank of the stiffness matrix (computed using
        :func:`numpy.linalg.matrix_rank`, which internally uses singular value
        decomposition for numerical stability) instead of computing its
        determinant, which is computationally expensive and potentially
        unstable for large systems. If the rank of the stiffness matrix is
        smaller than its dimension, the matrix is singular, indicating a
        kinematic system or an incorrect system description [1]_. Using the
        rank in this way provides a robust and numerically stable check [2]_.

        References
        ----------
        .. [1] D. Dinkler. "Grundlagen der Baustatik: Modelle und
               Berechnungsmethoden für ebene Stabtragwerke". Band 1, 2011.

        .. [2] J. Dankert, H. Dankert (Hrsg.). "Mathematik für die
               Technische Mechanik". Online: http://www.tm-mathe.de/,
               abgerufen am 22.03.2025.
        """
        k, p = self.boundary_conditions
        matrix_shape = k.shape
        self.logger.info(
            f"Checking solvability of system matrix with shape {matrix_shape}")

        rank = np.linalg.matrix_rank(k)
        matrix_size = k.shape[0]

        self.logger.debug(f"Matrix rank: {rank}, Matrix size: {matrix_size}")

        if rank < matrix_size:
            self.logger.warning(
                f"System is not solvable (singular matrix). Rank: {rank}, "
                f"Size: {matrix_size}")
            return False
        else:
            self.logger.info(
                f"System is solvable. Rank: {rank}, Size: {matrix_size}")
            return True

    # SOLUTION -------------------------------------------------------
    # Node results
    @cached_property
    def node_deform(self):
        r"""Solves the linear system for the nodal deformations.

        This function computes the nodal displacement vector :math:`\Delta`
        by solving the linear system composed of the modified global stiffness
        matrix :math:`K_{mod}` and the modified global load vector
        :math:`P`, which are the result of applying the
        :py:attr:`boundary_conditions`.

        The resulting displacement vector contains the deformations of
        each node in its local coordinate system.

        Returns
        -------
        :any:`numpy.array`
            The displacement vector :math:`\Delta` containing the nodal
            deformations.

        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info("Calculating node deformations (node_deform)")
        modified_stiffness_matrix, modified_p = self.boundary_conditions
        self.logger.debug(
            f"Matrix shape: {modified_stiffness_matrix.shape}, "
            f"Vector shape: {modified_p.shape}")

        deformations = np.linalg.solve(modified_stiffness_matrix, modified_p)
        self.logger.info("Node deformation calculation completed")
        self.logger.debug(f"Node deformation: \n {repr(deformations)}")
        return deformations

    @cached_property
    def system_node_deform(self):
        r"""Calculation of the node deformation of the node element in the
        global coordinate system :math:`\tilde{P}^{supp}_{n}`.

        The node deformation :py:attr:`node_deform` are initially
        given in the respective nodal coordinate system. For the presentation
        of the results, it may be useful to transform the computed node
        deformation of node *n* into the global coordinate system using the
        transformation matrix :math:`T_{node}`.

        Returns
        -------
        :any:`numpy.array`
            A vector with dimensions (dof * number of nodes, 1) that contains
            the node deformation referenced to the global coordinate system.

        Notes
        -----
            The transformation of the node deformations into the global
            coordinate system is performed as follows:

            .. math:: \tilde{\Delta}_{n} = T_{node} \cdot \Delta_{n}

            The transformation is only necessary if the node is rotated.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info(
            "Transformation of the node deformation into the global "
            "coordinate system")
        node_attribute = deepcopy(self.node_deform)
        transformed_nodes = 0

        for idx, node in enumerate(self.nodes):
            if node.rotation != 0:
                self.logger.debug(
                    f"Transforming node deformation for node {idx + 1} "
                    f"with rotation {node.rotation} radians")
                node_attribute[idx * self.dof: (idx + 1) * self.dof, :] = (
                        transformation_matrix(-node.rotation) @
                        node_attribute[idx * self.dof: (idx + 1) * self.dof, :]
                )
                transformed_nodes += 1

        self.logger.info(f"Transformation completed "
                         f"(transformed {transformed_nodes} nodes)")
        self.logger.debug(f"Node Deformation in the global coordinate "
                          f"system: \n"
                          f" {repr(node_attribute)}")
        return node_attribute

    @cached_property
    def node_support_forces(self):
        r"""Calculation of the support reaction of the node element
        :math:`P^{supp}`.

        To calculate the support reactions, the computed node displacements
        for all nodes :math:`\Delta` from the attribute :py:attr:`node_deform`,
        the bar end forces due to the load :py:attr:`f0`, and the nodal loads
        :py:attr:`p0` are used.

        Returns
        -------
        :any:`numpy.array`
            A vector with dimensions (dof * number of nodes, 1) containing the
            support reactions in the nodal coordinate system.

        Notes
        -----
            The formula for calculating the support reactions is as follows:

            .. math:: P^{supp} = K \cdot \Delta + F^{(0)} - P^{(0)}

            In the calculation algorithm, the deformations from the elastic
            supports are also taken into account.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info("Calculating support forces in the nodal coordinate "
                         "system")
        elastic_vec = np.vstack(np.diag(self.elastic_matrix))

        support_forces = (self.system_matrix @ self.node_deform + self.f0
                          - self.p0 - elastic_vec * self.node_deform)

        self.logger.info("Support force calculation completed")
        self.logger.debug(f"Support force: \n {repr(support_forces)}")
        return support_forces

    @cached_property
    def system_support_forces(self):
        r"""Calculation of the support reaction of the node element in the
        global coordinate system :math:`\tilde{P}^{supp}_{n}`.

        The support reactions :py:attr:`node_support_forces` are initially
        given in the respective nodal coordinate system. For the presentation
        of the results, it may be useful to transform the computed support
        reaction of node *n* into the global coordinate system using the
        transformation matrix :math:`T_{node}`.

        Returns
        -------
        :any:`numpy.array`
            A vector with dimensions (dof * number of nodes, 1) that contains
            the support reactions referenced to the global coordinate system.

        Notes
        -----
            The transformation of the support reactions into the global
            coordinate system is performed as follows:

            .. math:: \tilde{P}^{supp}_{n} = T_{node} \cdot P^{supp}_{n}

            The transformation is only necessary if the node is rotated.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info(
            "Transformation of the support forces into the global coordinate "
            "system")
        node_attribute = deepcopy(self.node_support_forces)
        transformed_nodes = 0

        for idx, node in enumerate(self.nodes):
            if node.rotation != 0:
                self.logger.debug(
                    f"Transforming support forces for node {idx + 1} "
                    f"with rotation {node.rotation} radians")
                node_attribute[idx * self.dof: (idx + 1) * self.dof, :] = (
                        transformation_matrix(-node.rotation) @
                        node_attribute[idx * self.dof: (idx + 1) * self.dof, :]
                )
                transformed_nodes += 1

        self.logger.info(f"Transformation completed "
                         f"(transformed {transformed_nodes} nodes)")
        self.logger.debug(f"Support force in the global coordinate system: \n"
                          f" {repr(node_attribute)}")
        return node_attribute

    # Bar results
    @cached_property
    def bar_deform(self):
        """Computes the deformation vectors for each bar in the local bar
        coordinate system.

        For each bar in the system, this function extracts the nodal
        deformations of its two connected nodes (each with 3 DOFs), stacks them
        into a (6, 1) array, and transforms them from the node coordinate
        system into the local bar coordinate system using the bar's
        transformation matrix.

        The result is a list of deformation vectors that describe how each bar
        deforms within its own local coordinate system.

        Returns
        -------
        :any:`list` of numpy.ndarray
            A list of (6, 1) arrays, each representing the bar deformation in
            the local bar coordinate system.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info(
            "Map the node deformation to the bar and transform them into the "
            "bar coordinate system")
        deform = self.node_deform
        dof = self.dof
        total_bars = len(self.mesh)

        deformations = []
        for bar_idx, bar in enumerate(self.mesh):
            self.logger.debug(f"Processing bar {bar_idx + 1}/{total_bars}")
            bar_deform = np.transpose(bar.transformation_matrix()) @ np.vstack(
                [
                    deform[self.nodes.index(bar.node_i) * dof:
                           self.nodes.index(bar.node_i) * dof + dof],
                    deform[self.nodes.index(bar.node_j) * dof:
                           self.nodes.index(bar.node_j) * dof + dof]
                ])
            deformations.append(bar_deform)

        self.logger.info(
            f"Bar deformation calculation completed for {total_bars} bars")
        return deformations

    @cached_property
    def internal_forces(self):
        r"""Calculates the internal forces of the statical system.

        To compute the internal forces at the ends of each bar element in the
        local coordinate system, the bar-end deformations :py:attr:`bar_deform`
        must be applied to the element stiffness relations.

        Returns
        -------
        :any:`list` of numpy.ndarray
            A list of (6, 1) arrays, each representing the internal forces in
            the local bar coordinate system.

        Notes
        -----
            The result of this equation yields the internal forces at the bar
            ends in the local coordinate system:

            .. math:: f^{'} = k^{'} \cdot \delta^{'} + f^{(0)'}
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info("Calculating internal forces for all bars")
        bar_deform = self.bar_deform
        total_bars = len(self.mesh)

        forces = []
        for bar_idx, (bar, deform) in enumerate(zip(self.mesh, bar_deform)):
            self.logger.debug(
                f"Processing internal forces for bar "
                f"{bar_idx + 1}/{total_bars}")
            internal_force = bar.stiffness_matrix(
                to_node_coord=False) @ deform + bar.f0(
                to_node_coord=False) + bar.f0_point
            forces.append(internal_force)

        self.logger.info(
            f"Internal force calculation completed for {total_bars} bars")
        return forces

    @cached_property
    def bar_deform_hinge(self):
        r"""Computes the relative deformations caused by hinges.

        Since various types of hinges can be placed at the ends of a bar, it is
        essential to consider their influence in the deformation analysis. The
        presence of a hinge leads to a discontinuous deformation at the
        corresponding node.

        Returns
        -------
        :any:`list` of numpy.ndarray
            A list of (6×1) vectors representing the relative displacements of
            the nodes for each bar in the system.

        Notes
        -----
            The algorithm iterates through all bars in the system and checks
            for the presence of hinges. If a hinge is found, the corresponding
            modified stiffness matrix :math:`k^{'}` and modified initial force
            vector :math:`f^{(0)'}` are used to calculate the bar's deformation
            :math:`\delta^{'}`. This step is essential for determining the
            relative deformations introduced by the hinges.

            The internal bar forces are given by:

            .. math::
                f^{'} = k^{'} \cdot \delta^{'} + f^{(0)'}

            When a hinge is present, the deformation at the node is no longer
            equal to the bar deformation. For instance, in the case of a
            rotational hinge at the end of a bar, the total rotation is the sum
            of the known nodal rotation :math:\varphi_{j}^{n} and the relative
            rotation :math:`\Delta \varphi_{j}`:

            .. math::
                \varphi_{j} = \varphi_{j}^{n} + \Delta \varphi_{j}

            The nodal deformation :math:`\delta^{(n)'}` is already known via
            the attribute :py:attr:`bar_deform`. The algorithm then calculates
            the relative deformation introduced by the hinge. To achieve this,
            the stiffness matrix and force vector are reduced to retain only
            the degrees of freedom associated with the hinge. This reduction
            allows the computation of the relative displacement at each
            hinge location.

            The calculation of internal forces is then extended to incorporate
            both known nodal and relative deformations:

            .. math::
                f^{'} = k^{'} \cdot (\delta^{(n)'} + \Delta \delta^{'}) \
                        + f^{(0)'}


            From this relationship, a linear system of equations is derived:

            .. math::
                A \cdot x = b

            where:

            .. math::
                A = k'{red_n}
                b = -k'{red} \cdot \delta^{(n)'} - f^{(0)'}

            The solution vector :math:x represents the total relative
            deformations caused by the hinges.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info(
            "Calculating bar deformations from hinge modifications")
        deform_list = []
        bar_deform_list = self.bar_deform
        total_bars = len(self.mesh)
        bars_with_hinges = 0

        for i, bar in enumerate(self.mesh):
            self.logger.debug(
                f"Processing bar {i + 1}/{total_bars} for hinge analysis")
            delta_rel = np.zeros((6, 1))
            if True in bar.hinge:
                self.logger.debug(f"Found hinge at bar {i + 1}")
                bars_with_hinges += 1

                k = bar.stiffness_matrix(hinge_modification=False)
                bar_deform = bar_deform_list[i]
                f0 = bar.f0(hinge_modification=False)

                idx = [i for i, value in enumerate(bar.hinge) if value]
                if idx:
                    self.logger.debug(f"Solving for hinge DOFs: {idx}")
                    rhs = k[np.ix_(idx, range(6))] @ bar_deform + f0[idx]
                    delta_rel[idx] = (
                        np.linalg.solve(k[np.ix_(idx, idx)], -rhs))

            deform_list.append(delta_rel)

        self.logger.info(
            f"Hinge deformation analysis completed ({bars_with_hinges} "
            f"bars with hinges)")
        return deform_list

    @cached_property
    def bar_deform_displacements(self):
        """Transforms the support-induced nodal displacements into local
        bar-end displacements.

        Using the transformation matrix of each bar, the nodal displacements
        caused by support stresses are rotated from the nodal coordinate system
        into the local bar coordinate system.

        Returns
        -------
        :any:`list` of numpy.ndarray
            A list of (6×1) vectors representing the local end displacements
            of each bar in the system resulting from nodal displacement inputs.

        See Also
        --------
        :any:`NodeDisplacement`
            For nodal displacement values in the global coordinate system.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info(
            "Calculating bar deformations from node displacements")
        total_bars = len(self.mesh)

        deformations = []
        for bar_idx, bar in enumerate(self.mesh):
            self.logger.debug(
                f"Transforming displacements for bar "
                f"{bar_idx + 1}/{total_bars}")
            bar_deform = np.transpose(bar.transformation_matrix()) @ np.vstack(
                (
                    bar.node_i.displacement, bar.node_j.displacement))
            deformations.append(bar_deform)

        self.logger.info(
            f"Nodal displacement transformation completed for {total_bars} "
            f"bars")
        return deformations

    @cached_property
    def bar_deform_list(self):
        """Combines all deformation contributions for each bar, resulting in
        the total deformation at the bar ends in the local coordinate system.

        This method adds the deformations from three different sources:
            * Deformation due to hinges (:py:attr:`hinge_modifier`)
            * Internal deformation from structural analysis \
            (:py:attr:`bar_deform`)
            * Displacement-induced deformation from nodal support movements \
            (:py:attr:`bar_deform_displacements`)

        Returns
        -------
        :any:`list` of numpy.ndarray
            A list of (6×1) vectors representing the total end deformations
            of each bar in the system, expressed in the local bar coordinate
            system.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        self.logger.info("Combining all deformation contributions")
        total_bars = len(self.mesh)

        combined_results = []
        for i in range(total_bars):
            self.logger.debug(
                f"Combining deformation components for bar "
                f"{i + 1}/{total_bars}")
            result = (self.bar_deform_hinge[i] + self.bar_deform[i] +
                      self.bar_deform_displacements[i])
            combined_results.append(result)

        self.logger.info(
            f"Combined deformation calculation completed for {total_bars} "
            f"bars")
        return combined_results

    @cached_property
    def system_deform_list(self):
        """Transforms the bar deformations for each bar into the
        system deformations.

        Returns
        -------
        :any:`list` of numpy.ndarray
            A list of (6, 1) arrays, each representing the deformation in
            the system coordinate system.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        bar_deform = self.bar_deform_list
        return [
            bar.transformation_matrix(False) @ deform
            for bar, deform in zip(self.mesh, bar_deform)
        ]

    @cached_property
    def node_deform_list(self):
        """Constructs a list of nodal deformation arrays for each bar in the
        system.

        This function assembles the nodal displacement vectors for each
        individual bar by extracting the corresponding deformation values from
        the global displacement vector. Each bar connects two nodes, and each
        node has three degrees of freedom (e.g., displacement in x and z
        directions, and rotation). Therefore, each array in the resulting list
        has a shape of (6, 1), representing the deformations associated with
        both nodes of the bar. The deformations are given in the local node
        coordinate system.

        Returns
        -------
        :any:`list` of numpy.ndarray
            A list of (6, 1) arrays, one for each bar, containing the nodal
            deformations (3 DOFs per node) in the node coordinate system.
        """
        if not self.solvable:
            raise ValueError(
                "The linear system is not solvable. "
                "The stiffness matrix is singular or poorly conditioned. "
                "Check the supports, hinges, or the overall system definition."
            )
        deform = self.node_deform
        dof = self.dof
        return [
            np.vstack([
                deform[self.nodes.index(bar.node_i) * dof:
                       self.nodes.index(bar.node_i) * dof + dof],
                deform[self.nodes.index(bar.node_j) * dof:
                       self.nodes.index(bar.node_j) * dof + dof]
            ])
            for bar in self.mesh
        ]
