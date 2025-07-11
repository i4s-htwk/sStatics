
from dataclasses import dataclass
from typing import Literal

from sstatics.core.postprocessing.calc_methods.equation_of_work import (
    EquationOfWork
)
from sstatics.core.postprocessing.results import SystemResult
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System, SystemModifier
from sstatics.core.solution.first_order import FirstOrder


@dataclass(eq=False)
class PVK:
    """Executes the principle of virtual forces (PVK).

    This class allows the application of the principle of virtual forces
    (German: Prinzip der virtuellen Kräfte) to a structural system by
    applying virtual loads and computing the resulting work.

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
            self, obj: Node, force: Literal['fx', 'fz', 'fm'],
            virt_force: float = 1):
        """Adds a virtual load to a node.

        Applies a virtual load to a given node. For the method to be valid,
        only one virtual load should be applied across the system. All
        previous loads are deleted before the virtual load is applied.

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
            Magnitude of the virtual load.
        """
        self.modifier.delete_loads()
        self.modifier.modify_node_force_vir(obj, force, virt_force)

    def add_virtual_bar_load(
            self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
            virt_force: float = 1, position: float = 0):
        """Adds a virtual load to a bar.

        Applies a virtual load to a given bar. All existing loads in the
        system are removed first. If the load position is between 0 and 1,
        the bar is split in the original system.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar to which the virtual load is applied.
        force : {'fx', 'fz', 'fm'}
            Direction of the applied virtual load:
                * 'fx': Global x-direction
                * 'fz': Global z-direction
                * 'fm': Moment at the bar
        virt_force : float, default=1
            Magnitude of the virtual load.
        position : float, default=0
            Position along the bar (0 to 1) where the virtual load is applied.
                """
        self.modifier.delete_loads()
        self.modifier.modify_bar_force_vir(obj, force, virt_force, position)
        # self.system.mesh({obj: [position]})

    def calc(self):
        """Performs the calculation using the principle of virtual forces.

        Returns
        -------
        :any:`float`
            The internal virtual work due to the applied virtual and real
            systems.
        """
        s1 = SystemResult(self.system, *FirstOrder(self.system).calc)
        s2 = SystemResult(self.modifier.system,
                          *FirstOrder(self.modifier.system).calc)
        return EquationOfWork(s1, s2).delta_s1_s2()

    def show_original_bars(self, bar):
        """Placeholder for visualizing original bars.

        Parameters
        ----------
        bar : :any:`Bar`
            The bar to be visualized.
        """
        return


class RED(PVK):
    """Executes the reduction theorem.

    This class implements the reduction theorem, which simplifies statically
    indeterminate systems by selectively removing constraints to reach
    determinacy. Then, virtual loads are applied for calculation.

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the reduction theorem.
    """

    def __post_init__(self):
        super().__post_init__()
        self.base_modified_system = None  # Modifiziertes System -> HS

    def modify_node(self, obj: Node, support: Literal['u', 'w', 'phi']):
        """Modifies the support conditions of a node.

        Frees the specified degree of freedom at the given node. Can be
        executed multiple times to reach a statically determinate system.

        Parameters
        ----------
        obj : :any:`Node`
            Node whose support conditions are to be modified.
        support : {'u', 'w', 'phi'}
            Degree of freedom to be released:
                * 'u': Horizontal
                * 'w': Vertical
                * 'phi': Rotation
        """
        self.modifier.modify_support(obj, support)
        self.base_modified_system = self.modifier.system

    def modify_bar(
            self, obj: Bar,
            hinge: Literal[
                'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
                'hinge_u_j', 'hinge_w_j', 'hinge_phi_j'
            ]
    ):
        """Inserts a hinge into a bar.

        Applies a hinge at the specified location on the bar to release the
        corresponding internal constraint.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar to which a hinge is applied.
        hinge : {'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
                 'hinge_u_j', 'hinge_w_j', 'hinge_phi_j'}
            Specifies the hinge and its location (i or j node).
        """
        self.modifier.insert_hinge(obj, hinge)
        self.base_modified_system = self.modifier.system

    def delete_bar(self, obj: Bar):
        """Deletes a bar from the system.

        Removes the specified bar from the structural system.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar to be removed.
        """
        self.modifier.delete_bar(obj)
        self.base_modified_system = self.modifier.system

    def degree_of_static_indeterminacy(self):
        """Calculates the degree of static indeterminacy.

        Returns
        -------
        :any:`int`
            The degree of static indeterminacy of the system.
        """
        support = sum((n.u != 'free') + (n.w != 'free') + (n.phi != 'free')
                      for n in self.system.nodes())
        hinge = sum(sum(h is True for h in b.hinge) for b in self.system.bars)
        return support + 3 * len(self.system.bars) - (
            3 * len(self.system.nodes(False)) + hinge)

    def calc(self):
        """Performs the calculation using the reduction theorem.

        Raises
        ------
        ValueError
            If the system is not statically determinate or if no modified
            system is defined.

        Returns
        -------
        :any:`float`
            The internal virtual work due to the base and virtual systems.
        """
        if self.degree_of_static_indeterminacy() != 0:
            raise ValueError(f'The system is statically indeterminate \
            (degree: {self.degree_of_static_indeterminacy()}).')

        if not self.base_modified_system:
            raise ValueError('There has to be a modified system.')

        base_result = SystemResult(self.base_modified_system,
                                   *FirstOrder(self.base_modified_system).calc)
        virt_result = SystemResult(self.modifier.system,
                                   *FirstOrder(self.modifier.system).calc)
        return EquationOfWork(base_result, virt_result).delta_s1_s2()


class KGV(RED):

    def ESZ(self):
        """Calculates the ESZ result systems.

        Placeholder for ESZ system computation.
        """
        return

    def add_virtuell_node_load(
            self, obj: Node, force: Literal['fx', 'fz', 'fm'],
            virt_force: float = 1):
        """Not allowed in this method.

        Raises
        ------
        ValueError
            If a virtual load is attempted to be applied.
        """
        super().add_virtual_node_load(obj, force, virt_force)
        raise ValueError('With this calculation method, no virtual load can be'
                         'applied.')

    def add_virtuell_bar_load(
            self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
            virt_force: float = 1, position: float = 0):
        """Not allowed in this method.

        Raises
        ------
        ValueError
            If a virtual load is attempted to be applied.
        """
        super().add_virtual_bar_load(obj, force, virt_force, position)
        raise ValueError('With this calculation method, no virtual load can be'
                         'applied.')

    def vorzahlen(self):
        return

    def belastungszahlen(self):
        return

    def ueberzaehlige(self):
        """Solves the system of equations.

        Placeholder for solving Ax = b to obtain redundant forces.
        """
        return
