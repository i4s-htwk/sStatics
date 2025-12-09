
from copy import deepcopy
from dataclasses import dataclass, field

from typing import Literal

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.calc_methods import PVF


@dataclass(eq=False)
class ReductionTheorem(PVF):
    """Executes the Reduction Theorem.

    This class applies the Reduction Theorem (german: Reduktionssatz) to
    simplify statically indeterminate systems by removing constraints to
    achieve a statically determinate system, then applying virtual loads for
    analysis.

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the Reduction Theorem.
    """

    _released_system: System = field(init=False, default=None)
    __virt_system: System = field(init=False, default=None)

    def __post_init__(self):
        """Initialize reduction workflow and state.

        Calls the base initializer and prepares an internal snapshot of
        the modifier after each release operation to track the reduced
        configuration.
        """
        self._system_modifier = SystemModifier(system=self.system)
        self._released_modifier = None
        self.logger.info("Reduction Theorem successfully created.")

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
                * :python:`u`: Horizontal (x-direction)
                * :python:`w`: Vertical (z-direction)
                * :python:`phi`: Rotation

        Raises
        ------
        ValueError
            If support type is invalid.
        """
        self.logger.info(
            f"Modify node support: Node={obj}, Support={support}."
        )

        if support not in ['u', 'w', 'phi']:
            msg = (f"Support component must be 'u', 'w' or 'phi'. "
                   f"Got {support}.")
            self.logger.error(msg)
            raise ValueError(msg)

        # Apply the node modification
        self._system_modifier.modify_support(obj, support)
        self.logger.debug("Node modification has been applied successfully.")

        # Update modifier for released system
        self._released_modifier = deepcopy(self._system_modifier)
        self._released_system = self._released_modifier.system
        self.logger.info("A copy of the system modifier has been assigned to "
                         "the released modifier.")

    def modify_bar(
            self, obj: Bar,
            hinge: Literal[
                'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
                'hinge_u_j', 'hinge_w_j', 'hinge_phi_j']):
        """Inserts a hinge at a specified location in a bar.

        Inserts a specified hinge at a bar. Multiple
        calls can be made to insert additional hinges until the system becomes
        statically determinate. Hierbei ist es wichtig, dass das System nicht
        verschieblich wird

        Parameters
        ----------
        obj : :any:`Bar`
            The bar where the hinge will be inserted.
        hinge : {'hinge_u_i', 'hinge_w_i', 'hinge_phi_i','hinge_u_j', \
                'hinge_w_j', 'hinge_phi_j'}
            The hinge to be applied and its location: Anfangsknoten (node_i )
            oder Endknoten (node_j).

        Raises
        ------
        ValueError
            If hinge type is invalid.
        """
        self.logger.info(f"Insert hinge at: Bar={obj}, hinge_type={hinge}.")

        if hinge not in ['hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
                         'hinge_u_j', 'hinge_w_j', 'hinge_phi_j']:
            msg = (
                f"Hinge type must be 'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',"
                f" 'hinge_u_j', 'hinge_w_j' or 'hinge_phi_j'. Got {hinge}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Apply the bar modification
        self._system_modifier.insert_hinge(obj, hinge)
        self.logger.debug("Bar modification applied successfully.")

        # Update modifier for released system
        self._released_modifier = deepcopy(self._system_modifier)
        self._released_system = self._released_modifier.system
        self.logger.info("A copy of the system modifier has been assigned to "
                         "the released modifier.")

    def delete_bar(self, obj: Bar):
        """Deletes a bar from the structural system.

        Um ein statisch bestimmtes System zu erhalten, ist es ebenfalls möglich
        Stäbe des modellierten Systems zu löschen. Eine Überprüfung der
        Verformungsanteile für eine mögliche Stablöschung wird intern nicht
        durchgeführt.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar to be removed from the system.
        """
        self.logger.info(f"Delete bar {obj}")

        self._system_modifier.delete_bar(obj)
        self.logger.debug("Bar has been deleted successfully.")

        # Update modifier for released system
        self._released_modifier = deepcopy(self._system_modifier)
        self._released_system = self._released_modifier.system
        self.logger.info("A copy of the system modifier has been assigned to "
                         "the released modifier.")

    @property
    def released_system(self):
        """Returns the current released system.

        Returns
        -------
        :any:`System`
            The reduced (statically determinate) system.
        """
        if self._released_system is None:
            raise ValueError("A released system must be defined.")
        return self._released_system

    @property
    def degree_of_static_indeterminacy(self):
        """Calculates the degree of static indeterminacy of the system.

        This function checks the static determinacy of the currently modified
        system. A system is statically determinate if n = 0.
        It is statically indeterminate if n > 0 and statically unstable
        if n < 0.

        Returns
        -------
        :any:`int`
            The number of redundant constraints in the system.
        """
        self.logger.info(
            "Calculating the degree of static indeterminacy of "
            "the modified system.")

        support = sum((n.u != 'free') + (n.w != 'free') + (n.phi != 'free')
                      for n in self._system_modifier.system.nodes())
        self.logger.debug(f"Total number of support reactions: {support}")

        hinge = sum(sum(h is True for h in b.hinge)
                    for b in self._system_modifier.system.bars)
        self.logger.debug(f"Total number of hinges: {hinge}")

        n = support + 3 * len(self._system_modifier.system.bars) - (
                3 * len(self._system_modifier.system.nodes('bars')) + hinge)
        self.logger.debug(f"Degree of static indeterminacy: {n}")

        return n

    def add_virtual_node_load(
            self, obj: Node,
            force: Literal['fx', 'fz', 'fm'], virt_force: float = 1):
        self._validate_released_system()
        super().add_virtual_node_load(obj, force, virt_force)

    def add_virtual_bar_load(
            self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
            position: float = 0, virt_force: float = 1):
        self._validate_released_system()
        super().add_virtual_bar_load(obj, force, position, virt_force)

    def add_virtual_moment_couple(
            self,
            bar_positive_m: Bar,
            bar_negative_m: Bar,
            connecting_node: Node,
            virt_force: float = 1.0):
        self._validate_released_system()
        super().add_virtual_moment_couple(
            bar_positive_m, bar_negative_m, connecting_node, virt_force)

    def plot_released_system(self):
        if self.released_system is None:
            msg = "No released system is defined."
            self.logger.error(msg)
            raise ValueError(msg)
        from sstatics.graphic_objects.system import SystemGraphic
        return SystemGraphic(self.released_system).show()

    def _validate_released_system(self):
        """Validate that the system is released and statically determinate.

        Raises
        ------
        ValueError
            If the system is still statically indeterminate or if no released
            system has been defined.
        """
        self.logger.info(
            "Validating the released system to ensure it is statically "
            "determinate and that a primary system has been modified.")
        if self.degree_of_static_indeterminacy != 0:
            raise ValueError(
                f"The system is statically indeterminate "
                f"(degree {self.degree_of_static_indeterminacy})."
            )

        if not self._released_system:
            raise ValueError("A released system must be defined.")
