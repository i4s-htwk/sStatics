from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from sstatics.core.postprocessing.calc_methods.equation_of_work import (
    EquationOfWork
)
from sstatics.core.postprocessing.results import SystemResult
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System
from sstatics.core.solution.first_order import FirstOrder


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
        self.modifier = SystemModifier(system=self.system)

    def add_virtual_node_load(
            self, obj: Node, force: Literal['fx', 'fz', 'fm']):
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
        """
        self.modifier.delete_loads()
        self.modifier.modify_node_force_vir(obj, force)

    def add_virtual_bar_load(
            self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
            position: float = 0):
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
        """
        self.modifier.delete_loads()
        self.modifier.modify_bar_force_vir(obj, force, position=position)

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
        return SystemResult(system=system,
                            bar_deform_list=solve.bar_deform_list,
                            bar_internal_forces=solve.internal_forces,
                            node_deform=solve.node_deform,
                            node_support_forces=solve.node_support_forces,
                            system_support_forces=solve.system_support_forces
                            )

    def calc(self):
        """Performs the calculation using the Principle of Virtual Forces.

        Before calculations are made, both the base and virtual systems must
        have consistent mesh divisions.

        Returns
        -------
        :any:`numpy.array`
            The internal virtual work due to the applied virtual and real
            systems.
        """
        self.modifier.system.create_mesh(
            self.modifier.division_positions_mesh())
        self.system.create_mesh(self.modifier.memory_bar_point_load)
        s1 = self._get_result(self.system)
        s2 = self._get_result(self.modifier.system)
        return EquationOfWork(s1, s2).delta_s1_s2()


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
        hinge : {'hinge_u_i', 'hinge_w_i', 'hinge_phi_i', \
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
        self.released_modifier = self.modifier

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

    def calc(self):
        """Performs the calculation using the Reduction Theorem.

        Before calculations are made, both the released system
        and virtual systems must have consistent mesh divisions.

        Raises
        ------
        ValueError
            If the system is still statically indeterminate or if no
            modified system has been defined.

        Returns
        -------
        :any:`float`
            The internal virtual work computed from the reduced and virtual
            systems.
        """
        if self.degree_of_static_indeterminacy() != 0:
            raise ValueError(
                f'The system is statically indeterminate degree: '
                f'({self.degree_of_static_indeterminacy()}).')

        if not self.released_modifier:
            raise ValueError('A modified system must be defined.')

        self.released_modifier.system.create_mesh(
            self.released_modifier.division_positions_mesh())
        self.modifier.system.create_mesh(
            self.modifier.division_positions_mesh())

        hs_results = self._get_result(self.released_modifier.system)
        virt_result = self._get_result(self.modifier.system)
        return EquationOfWork(hs_results, virt_result).delta_s1_s2()


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
        """Returns the released system in the real load state (RLS(.

        Returns
        -------
        :any:`System`
            The reduced primary system.
        """
        return self.released_modifier.system

    def _get_uls_systems(self):
        """Returns a list of virtual systems for the unit load states (ULS)
        used for each redundant.

        Returns
        -------
        list[:any:`System`]
            List of systems generated for each unit load case.
        """
        self.modifier.delete_loads()
        return self.modifier.create_uls_systems()

    def add_virtual_node_load(self, *args, **kwargs):
        raise ValueError("Virtual node loads are not allowed for this method.")

    def add_virtual_bar_load(self, *args, **kwargs):
        raise ValueError("Virtual bar loads are not allowed for this method.")

    def _mesh_uls(self):
        uls = []
        for ul in self._get_uls_systems():
            ul.create_mesh(self.modifier.division_positions_mesh())
            uls.append(ul)
        return uls

    def _mesh_rls_system(self):
        rls = self._get_rls_system()
        rls.create_mesh(self.modifier.division_positions_mesh())
        return rls

    def vorzahlen(self):
        """
        Calculates preliminary coefficients.

        Placeholder for actual implementation.
        """
        return

    def belastungszahlen(self):
        """
        Calculates the load coefficients (influence numbers).

        Placeholder for actual implementation.
        """
        return

    def ueberzaehlige(self):
        """
        Solves the system of equations to compute redundant forces.

        Placeholder for solving Ax = b for redundants.
        """
        return
