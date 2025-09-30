from copy import deepcopy
from dataclasses import dataclass
from typing import Literal
from functools import cached_property
import numpy as np
import logging
from tabulate import tabulate

from sstatics.core.postprocessing.calc_methods.equation_of_work import (
    EquationOfWork
)
from sstatics.core.postprocessing.results import SystemResult
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System
from sstatics.core.solution.first_order import FirstOrder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PVK:
    """Executes the Principle of Virtual Forces (PVK).

    This class allows the application of the Principle of Virtual Forces
    (German: Prinzip der virtuellen Kräfte) to a structural system by
    applying virtual loads and computing the resulting internal work.

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the principle of
        virtual forces.
    """

    system: System

    def __post_init__(self):
        """Initialize helper objects needed for PVK operations.

        Sets up a :any:`SystemModifier` for the provided system. The
        modifier manages load deletion, virtual load application and
        mesh related tasks to ensure consistent PVK evaluation.
        """
        self.modifier = SystemModifier(system=self.system)

    def add_virtual_node_load(
            self, obj: Node,
            force: Literal['fx', 'fz', 'fm'], virt_force: float = 1):
        # Entweder 1 oder -1
        """Adds a virtual load to a node.

        Applies a virtual load to a given node. To maintain the validity
        of the method, only one virtual load should be applied at a time.
        All previous loads are cleared before the new load is applied.

        Parameters
        ----------
        obj : :any:`Node`
            The node at which the virtual load is applied.
        force : {'fx', 'fz', 'fm'}
            Direction of the applied virtual load:
                * 'fx': Global x-direction
                * 'fz': Global z-direction
                * 'fm': Moment at the node
        virt_force : float, default=1
            Magnitude of the virtual force, default is 1.
        """
        self.modifier.delete_loads()
        self.modifier.modify_node_force_vir(obj, force, virt_force=virt_force)

    def add_virtual_bar_load(
            self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
            position: float = 0, virt_force: float = 1):
        """Adds a virtual load to a bar.

        Applies a virtual load at a specific location along a bar. All
        existing loads in the system are first removed. If the position is
        between 0 and 1, the bar is internally subdivided.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar to which the virtual load is applied.
        force : {'fx', 'fz', 'fm'}
            Direction of the applied virtual load:
                * 'fx': Global x-direction
                * 'fz': Global z-direction
                * 'fm': Moment at the bar
        position : float, default=0
            Relative position along the bar (from 0 to 1) where the virtual
            load is applied.
        virt_force : float, default=1
            Magnitude of the virtual force, default is 1.
        """
        self.modifier.delete_loads()
        self.modifier.modify_bar_force_vir(obj, force, position=position,
                                           virt_force=virt_force)

    def _get_result(self, system: System):
        """Computes the results for a given structural system.

        Parameters
        ----------
        system : :any:`System`
            The system to compute results for.

        Returns
        -------
        :any:`SystemResult`
            Result object containing internal forces and displacements.
        """
        solve = FirstOrder(system)

        if solve.solvable:
            return SystemResult(
                system=system,
                bar_deform_list=solve.bar_deform_list,
                bar_internal_forces=solve.internal_forces,
                node_deform=solve.node_deform,
                node_support_forces=solve.node_support_forces,
                system_support_forces=(
                    solve.system_support_forces
                ),
            )
        else:
            raise ValueError("System is kinematically instable")

    def deformation(self):
        """Performs the calculation using the Principle of Virtual Forces.

        Before calculations are made, both the base and virtual systems must
        have consistent mesh divisions.

        Returns
        -------
        :any:`np.array`
            The internal virtual work due to the applied virtual and real
            systems.
        """
        self.modifier.system.create_mesh(
            self.modifier.division_positions_mesh())
        self.system.create_mesh(self.modifier.memory_bar_point_load)
        s1 = self._get_result(self.system)
        s2 = self._get_result(self.modifier.system)
        return EquationOfWork(s1, s2).delta_s1_s2()

    def log_work_contributions(self, decimals: int = 6):
        """Logs bar wise and node wise PVK term contributions.

        Builds bar and node contribution matrices via :any:`EquationOfWork`
        and logs formatted tables including row and column sums.
        """
        self.modifier.system.create_mesh(
            self.modifier.division_positions_mesh())
        self.system.create_mesh(self.modifier.memory_bar_point_load)
        s1 = self._get_result(self.system)
        s2 = self._get_result(self.modifier.system)
        return EquationOfWork(s1, s2).log_work_contributions(decimals=decimals)


class RED(PVK):
    """Executes the Reduction Theorem.

    This class applies the Reduction Theorem to simplify statically
    indeterminate systems by removing constraints to achieve a statically
    determinate system, then applying virtual loads for analysis.

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the Reduction Theorem.
    """

    def __post_init__(self):
        """Initialize reduction workflow and state.

        Calls the base initializer and prepares an internal snapshot of
        the modifier after each release operation to track the reduced
        configuration.
        """
        super().__post_init__()
        self.released_modifier = None

    def modify_node(self, obj: Node, support: Literal['u', 'w', 'phi']):
        """Modifies the support conditions of a node.

        Releases a specified degree of freedom (DOF) at a node. Multiple
        calls can be made to release additional DOFs until the system becomes
        statically determinate.

        Parameters
        ----------
        obj : :any:`Node`
            Node whose support condition will be modified.
        support : {'u', 'w', 'phi'}
            Degree of freedom to release:
                * 'u': Horizontal (x-direction)
                * 'w': Vertical (z-direction)
                * 'phi': Rotation
        """
        self.modifier.modify_support(obj, support)
        self.released_modifier = deepcopy(self.modifier)

    def modify_bar(
            self, obj: Bar,
            hinge: Literal[
                'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
                'hinge_u_j', 'hinge_w_j', 'hinge_phi_j']):
        """Inserts a hinge at a specified location in a bar.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar where the hinge will be inserted.
        hinge : {'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
                 'hinge_u_j', 'hinge_w_j', 'hinge_phi_j'}
            The hinge to be applied and its location (node i or j).
        """
        self.modifier.insert_hinge(obj, hinge)
        self.released_modifier = deepcopy(self.modifier)

    def delete_bar(self, obj: Bar):
        """Deletes a bar from the structural system.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar to be removed from the system.
        """
        self.modifier.delete_bar(obj)
        self.released_modifier = deepcopy(self.modifier)

    def get_released_system(self):
        """Returns the current reduced system.

        Returns
        -------
        :any:`System`
            The reduced (statically determinate) system.
        """
        return self.released_modifier.system

    def degree_of_static_indeterminacy(self):
        """Calculates the degree of static indeterminacy of the system.

        Returns
        -------
        :any:`int`
            The number of redundant constraints in the system.
        """
        support = sum((n.u != 'free') + (n.w != 'free') + (n.phi != 'free')
                      for n in self.modifier.system.nodes())
        hinge = sum(sum(h is True for h in b.hinge)
                    for b in self.modifier.system.bars)
        return support + 3 * len(self.modifier.system.bars) - (
            3 * len(self.modifier.system.nodes('bars')) + hinge)

    def _validate_system_ready(self):
        """Validate that the system is released and statically determinate.

        Raises
        ------
        ValueError
            If the system is still statically indeterminate or if no released
            system has been defined.
        """
        if self.degree_of_static_indeterminacy() != 0:
            raise ValueError(
                f"The system is statically indeterminate "
                f"(degree {self.degree_of_static_indeterminacy()})."
            )

        if not self.released_modifier:
            raise ValueError("A released system must be defined.")

    def deformation(self):
        """Performs the calculation using the Reduction Theorem.

        First validates that the system is released and statically
        determinate (:meth:`_validate_system_ready`). Both the released
        and virtual systems must then have consistent mesh divisions.

        Raises
        ------
        ValueError
            If the system is still statically indeterminate or if no
            released system has been defined.

        Returns
        -------
        :any:`float`
            The internal virtual work computed from the reduced and virtual
            systems.
        """
        self._validate_system_ready()

        self.modifier.system.create_mesh(
            self.modifier.division_positions_mesh())
        self.system.create_mesh(
            self.modifier.memory_bar_point_load)
        s1 = self._get_result(self.system)
        s2 = self._get_result(self.modifier.system)
        return EquationOfWork(s1, s2).delta_s1_s2()

    def log_work_contributions(self, decimals: int = 6):
        """
        Logs detailed PVK work contributions for the reduced system.

        First validates that the system is released and statically
        determinate (:meth:`_validate_system_ready`). Then builds bar-wise
        and node-wise contribution matrices of the Equation of Work between
        the real system and the reduced system. The results are logged as
        formatted tables including row and column sums.

        Raises
        ------
        ValueError
            If the system is still statically indeterminate or if no
            released system has been defined.

        Returns
        -------
        None
            The method logs the results but does not return data.
        """
        self._validate_system_ready()

        self.modifier.system.create_mesh(
            self.modifier.division_positions_mesh())
        self.system.create_mesh(
            self.modifier.memory_bar_point_load)
        s1 = self._get_result(self.system)
        s2 = self._get_result(self.modifier.system)
        return EquationOfWork(s1, s2).log_work_contributions(decimals=decimals)


class KGV(RED):
    """
    Executes the Force Method (KGV)

    Method to calculate internal forces, support forces and deformations of
    statically undetermined systems (n > 0). This method is based on the
    principle of virtual forces (PvF).

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the Force Method.
    """

    def _get_rls_system(self):
        """Return the released system in the real load state (RLS).

        First validates that the system is released and statically
        determinate (:meth:`_validate_system_ready`).

        Returns
        -------
        :any:`System`
            The reduced primary system representing the real load state.
        """
        self._validate_system_ready()
        return self.released_modifier.system

    def _get_uls_systems(self):
        """Return virtual systems for the unit load states (ULS).

        First validates that the system is released and statically
        determinate (:meth:`_validate_system_ready`). All external
        loads are then cleared before generating the unit load
        systems, one for each redundant.

        Returns
        -------
        list[:any:`System`]
            List of generated systems, one per unit load case.
        """
        self._validate_system_ready()
        self.modifier.delete_loads()
        return self.modifier.create_uls_systems()

    def _mesh_uls_systems(self):
        """Meshes all ULS with divisions consistent to the modifier.

        Returns
        -------
        list[:any:`System`]
            List of meshed ULS systems in redundant order.
        """
        uls = []
        for uls_system in self._get_uls_systems():
            uls_system.create_mesh(
                self.modifier.division_positions_for(uls_system))
            uls.append(uls_system)
        return uls

    def _mesh_rls_system(self):
        """Meshes the RLS using the modifier division positions.

        Returns
        -------
        :any:`System`
            Meshed released system used as real load state.
        """
        rls = self._get_rls_system()
        rls.create_mesh(self.modifier.division_positions_mesh())
        return rls

    def add_virtual_node_load(self, *args, **kwargs):
        """Disallows virtual node loads for KGV.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        raise ValueError("Virtual node loads are not allowed for this method.")

    def add_virtual_bar_load(self, *args, **kwargs):
        """Disallows virtual bar loads for KGV.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        raise ValueError("Virtual bar loads are not allowed for this method.")

    def deformation(self):
        """Disallows computation of single deformations in KGV.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        raise ValueError("Deformation cannot be computed for this method.")

    @cached_property
    def uls_results(self):
        """
        Compute results for all unit load systems (ULS).

        Creates meshes for each ULS, solves them using first order
        analysis, and returns the corresponding result objects.

        Returns
        -------
        list[:any:`SystemResult`]
            List of results for all unit load systems in redundant order.
        """
        uls_systems_meshed = self._mesh_uls_systems()

        return [self._get_result(uls_system) for uls_system in
                uls_systems_meshed]

    def rls_results(self):
        """
        Compute the result of the released real load system (RLS).

        The released system is first meshed with consistent divisions,
        then analyzed using first order theory. The resulting object
        contains internal forces, deformations, and support reactions.

        Returns
        -------
        :any:`SystemResult`
            Result container for the released real load state.
        """
        rls_meshed = self._mesh_rls_system()

        return self._get_result(rls_meshed)

    def load_coef(self):
        """
        Calculate the load coefficients for the force method.

        Evaluates the Equation of Work between the released real load
        system (RLS) and each unit load system (ULS). The resulting
        coefficients form the right-hand side vector of the force method
        equation system.

        Returns
        -------
        :any:`np.ndarray`
            One-dimensional array of preliminary coefficients, with one
            entry per redundant.
        """
        return np.array([EquationOfWork(self.rls_results(), uls).delta_s1_s2()
                         for uls in self.uls_results])

    def influence_coef(self):
        """
        Calculate the influence number matrix for the force method.

        Evaluates the Equation of Work between all combinations of unit
        load systems (ULS). The resulting square matrix contains the
        coefficients that represent the coupling between redundants.
        This matrix is used as the system matrix in the force method.

        Returns
        -------
        :any:`np.ndarray`
            Two-dimensional square matrix of influence numbers, with one
            row and one column for each redundant.
        """
        return np.array([[EquationOfWork(uls_i, uls_j).delta_s1_s2()
                          for uls_j in self.uls_results]
                         for uls_i in self.uls_results])

    def redundants(self):
        """
        Solve the linear system to compute the redundant forces.

        Forms the system of equations A x = b, where A is the matrix of
        influence numbers (from :meth:`load_coef`) and b is the vector of
        preliminary coefficients (from :meth:`preliminary_coef`). Solving
        this system yields the values of the redundant forces.

        Returns
        -------
        :any:`np.ndarray`
            One-dimensional array containing the solved redundant forces.
        """
        return np.linalg.solve(self.influence_coef(), -self.load_coef())

    def log_work_contributions(self, decimals: int = 6):
        """
        Log detailed PVK contribution tables for RLS and all ULS pairings.

        Evaluates the Equation of Work for the released real load system
        (RLS) against each unit load system (ULS), as well as for all
        pairings of ULS among themselves. Each evaluation produces bar-
        wise and node-wise contribution tables, which are logged with
        identifiers delta_i0 for RLS–ULS pairings and delta_ik for ULS–
        ULS pairings.

        Returns
        -------
        None
            The method logs formatted tables but does not return data.
        """
        rls = self.rls_results()

        for i, uls_i in enumerate(self.uls_results, start=1):
            EquationOfWork(rls, uls_i).log_work_contributions(delta=f"{i}0")
            for k, uls_k in enumerate(self.uls_results, start=1):
                EquationOfWork(uls_i, uls_k).log_work_contributions(
                    delta=f"{i}{k}", decimals=decimals)

    def log_linear_system(self, decimals: int = 6):
        """
        Log the linear system of equations (A x = b) for the force method.

        Builds the system matrix from the influence numbers (A, obtained
        via :meth:`influence_coef`) and the preliminary coefficients
        (b, obtained via :meth:`load_coef`). The system matrix with the
        right-hand side is tabulated with labeled columns and logged.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places used for floating point formatting in
            the tabulated output. Default is ``6``.

        Returns
        -------
        None
            The method logs the linear system but does not return data.
        """
        A = self.influence_coef()
        b = self.load_coef().reshape(-1, 1)
        system_matrix = np.hstack([A, b])

        n = A.shape[0]
        headers = [f"X{i + 1}" for i in range(n)] + ["b"]

        table = tabulate(
            system_matrix,
            headers=headers,
            tablefmt="grid",
            floatfmt=f".{decimals}f"
        )

        logger.info("Linear system of equations (A x = b):\n%s", table)


class DMG(KGV):

    support_moment = KGV.redundants

    def log_linear_system(self, decimals: int = 6):
        """
        Log the linear system of equations (A x = b) for the force method.

        Builds the system matrix from the influence numbers (A, obtained
        via :meth:`influence_coef`) and the preliminary coefficients
        (b, obtained via :meth:`load_coef`). The system matrix with the
        right-hand side is tabulated with labeled columns and logged.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places used for floating point formatting in
            the tabulated output. Default is ``6``.

        Returns
        -------
        None
            The method logs the linear system but does not return data.
        """
        A = self.influence_coef() * 6
        b = self.load_coef().reshape(-1, 1) * 6
        system_matrix = np.hstack([A, b])

        n = A.shape[0]
        headers = [f"M{i + 1}" for i in range(n)] + ["b"]

        table = tabulate(
            system_matrix,
            headers=headers,
            tablefmt="grid",
            floatfmt=f".{decimals}f",
        )

        logger.info("Linear system of equations (A x = b):\n%s", table)
