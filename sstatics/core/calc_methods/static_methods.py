
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from functools import cached_property

import numpy as np
from tabulate import tabulate
from typing import Literal

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.postprocessing.equation_of_work import EquationOfWork
from sstatics.core.calc_methods.first_order import FirstOrder
from sstatics.core.logger_mixin import LoggerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PVF(LoggerMixin):
    """Executes the Principle of Virtual Forces (PvF).

    The Principle of Virtual Forces (german: Prinzip der virtuellen Kräfte)
    is used to compute displacements by applying a virtual load at the
    location and in the direction of the desired deformation. **Exactly one
    virtual load must be applied per deformation calculation**.

    This class provides tools to generate such a virtual system by:
        - applying a virtual load at a node,
        - applying a virtual load along a bar,
        - applying a virtual moment couple at a connecting node.

    After applying the virtual load, the class constructs:
        - the *virtual system* (with only the virtual load),
        - the *real system* (with the physical loads),
    ensures both systems have consistent meshing,
    and computes the displacement using the internal work equation
    (see :any:`EquationOfWork`).

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the principle of
        virtual forces.
    debug : :any:`bool`, default=False
        Enable debug logging for intermediate steps.

    """

    system: System
    debug: bool = False

    __virt_system: System = field(init=False, default=None)

    def __post_init__(self):
        """Initializes helper objects required for PVF operations.

        Creates a :class:`SystemModifier` which manages load deletion,
        application of virtual loads, and mesh adjustments to ensure that
        real and virtual systems remain consistent.
        """
        self._system_modifier = SystemModifier(system=self.system)
        self.logger.info("Principle of Virtual Forces successfully created.")

    def add_virtual_node_load(
            self, obj: Node,
            force: Literal['fx', 'fz', 'fm'], virt_force: float = 1):
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
                * :python:`fx`: system x-direction
                * :python:`fz`: system z-direction
                * :python:`fm`: Moment at the node
        virt_force : :any:`float`, default=1
            Magnitude of the virtual force, default is 1.
        """
        self.logger.info(
            f"Applying virtual node load: Node={obj}, Direction={force}, "
            f"Value={virt_force}"
        )

        self._validate_virtual_load(force)

        # Remove all previous loads
        self._system_modifier.delete_loads()
        self.logger.debug(
            "All existing loads have been deleted from the system.")

        # Apply the virtual node load
        self._system_modifier.modify_node_force_vir(
            obj, force, virt_force=virt_force)
        self.logger.debug("Virtual node load applied successfully.")

        # Update virtual system and load mode
        self.__virt_system = self._system_modifier.system
        self.logger.info("Virtual system created (node load).")

    def add_virtual_bar_load(
            self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
            position: float = 0, virt_force: float = 1):
        """Adds a virtual load to a bar.

        Applies a virtual load at a specific location along a bar. To maintain
        the validity of the method, only one virtual load should be applied at
        a time.All existing loads in the system are first removed. If the
        position is between 0 and 1, the bar is internally subdivided.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar to which the virtual load is applied.
        force : {'fx', 'fz', 'fm'}
            Direction of the applied virtual load:
                * :python:`fx`: system x-direction
                * :python:`fz`: system z-direction
                * :python:`fm`: Moment at the bar
        position : :any:`float`, default=0
            Relative position along the bar (from 0 to 1) where the virtual
            load is applied.
        virt_force : :any:`float`, default=1
            Magnitude of the virtual force, default is 1.
        """
        self.logger.info(
            f"Applying virtual bar load: Bar={obj}, Direction={force}, "
            f"Position={position}, Value={virt_force}"
        )

        self._validate_virtual_load(force)

        if not (0 <= position <= 1):
            msg = f"Position must be between 0 and 1. Got {position}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Remove all previous loads
        self._system_modifier.delete_loads()
        self.logger.debug(
            "All existing loads have been deleted from the system.")

        # Apply the virtual bar load
        self._system_modifier.modify_bar_force_vir(
            obj, force, position=position, virt_force=virt_force)
        self.logger.debug("Virtual bar load applied successfully.")

        # Update virtual system and load mode
        self.__virt_system = self._system_modifier.system
        self.logger.info("Virtual system created (bar load).")

    def add_virtual_moment_couple(
            self,
            bar_positive_m: Bar,
            bar_negative_m: Bar,
            connecting_node: Node,
            virt_force: float = 1.0):
        """Adds a virtual moment couple to two bars.

        A moment couple consists of equal and opposite virtual moments
        applied to two bars connected at the same node. All existing loads
        are removed beforehand.

        Parameters
        ----------
        bar_positive_m : :any:`Bar`
            Bar receiving the positive moment component.
        bar_negative_m : :any:`Bar`
            Bar receiving the negative moment component.
        connecting_node : :any:`Node`
            Node where the moment couple acts and to which both bars must be
            connected.
        virt_force : :any:`float`, default=1
            Magnitude of the moment, default is 1. The negative moment is
            applied as -virt_force.
        """
        self.logger.info(
            f"Applying virtual moment couple: +M on {bar_positive_m}, "
            f"-M on {bar_positive_m}, connecting node: {connecting_node},"
            f"Value={virt_force}"
        )
        # Validate that both bars share the connecting node
        bars_at_node = self.system.node_to_bar_map().get(connecting_node, [])
        if not (bar_positive_m in bars_at_node and bar_negative_m
                in bars_at_node):
            raise ValueError(
                "Both bars must be connected to the given connecting node."
            )

        # Determine moment application positions (0 = node_i, 1 = node_j)
        position_positive = 0 if (bar_positive_m.node_i ==
                                  connecting_node) else 1
        position_negative = 0 if (bar_negative_m.node_i ==
                                  connecting_node) else 1

        # Remove all previous loads
        self._system_modifier.delete_loads()
        self.logger.debug(
            "All existing loads have been deleted from the system.")

        # Apply the virtual bar load
        self._system_modifier.modify_bar_force_vir(
            bar_positive_m, 'fm', position=position_positive,
            virt_force=virt_force)

        self.logger.debug(
            "Virtual positive moment component applied successfully.")

        self._system_modifier.modify_bar_force_vir(
            bar_negative_m, 'fm', position=position_negative,
            virt_force=-virt_force)
        self.logger.debug(
            "Virtual negative moment component applied successfully.")

        # Update virtual system and load mode
        self.__virt_system = self._system_modifier.system
        self.logger.info("Virtual system created (moment couple).")

    @property
    def virtual_system(self):
        r"""Returns the structural system containing the applied virtual load.

        Returns
        -------
        :any:`System`
            The system with the virtual load

        Raises
        ------
        RuntimeError
            If no virtual load has been applied.
        """
        self._validate_virtual_system()
        return self.__virt_system

    @cached_property
    def solution_virtual_system(self):
        r"""Returns the first order analysis of the virtual system.

        A new :any:`FirstOrder` instance is created on first access, using
        the internally stored virtual system generated via
        :meth:`add_virtual_bar_load()` or :meth:`add_virtual_node_load()`.
        Before solving, the system is meshed consistently with the real system,
        ensuring matching integration lengths for the internal work
        calculation.

        Returns
        -------
        :any:`FirstOrder`
            FirstOrder instance operating on the virtual system.

        Raises
        ------
        RuntimeError
            If no virtual system exists.
        """
        self._validate_virtual_system()
        self.__virt_system.create_mesh(
            self._system_modifier.division_positions_mesh())
        self.logger.debug("Virtual system meshed for solution computation.")
        solution = FirstOrder(self.__virt_system)
        self.logger.debug("A FirstOrder instance has been created by using"
                          "the system with virtual loads.")
        return solution

    @cached_property
    def solution_real_system(self):
        r"""Returns the first order analysis of the system with real loads.

        A new :any:`FirstOrder` instance is created on first access, using
        the system with real loads.
        Before solving, the system is meshed consistently with the real system,
        ensuring matching integration lengths for the internal work
        calculation.

        Returns
        -------
        :any:`FirstOrder`
            FirstOrder instance operating on the system with real loads.

        Raises
        ------
        RuntimeError
            If no virtual load has been applied
            (mesh needs virtual division points).
        """
        self._validate_virtual_system()
        self.system.create_mesh(self._system_modifier.memory_bar_point_load)
        self.logger.debug("Real system meshed for solution computation.")
        solution = FirstOrder(self.system)
        self.logger.debug("A FirstOrder instance has been created by using"
                          "the system with real loads.")
        return solution

    def deformation(self):
        """Performs the calculation using the Principle of Virtual Forces.

        The deformation is evaluated exactly at the location and in the
        direction of the virtual load using the internal virtual work.

        Returns
        -------
        :any:`numpy.ndarray`
            The internal virtual work due to the applied virtual and real
            systems.
        """
        self.logger.info(
            "Passing the system with real loads and the system with virtual "
            "loads to the EquationOfWork class."
        )
        deformation = EquationOfWork(
            self.solution_real_system,
            self.solution_virtual_system).delta_ij

        self.logger.debug(f"Computed deformation value: {deformation}")
        return deformation

    def plot(
            self, mode: Literal['real', 'virt'] = 'real',
            kind: Literal[
                'normal', 'shear', 'moment', 'u', 'w', 'phi',
                'bending_line'] = 'normal',
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            result_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: int | None = None, n_disc: int = 10
    ):
        r"""Plot internal forces or deformation results of either the system
        with real or virtual loads.

        Parameters
        ----------
        mode : {'real', 'virt'}
            Defines whether the results of the system with real loads or the
            system with virtual loads should be plotted.
        kind : {'normal', 'shear', 'moment', 'u', 'w', 'phi', \
                'bending_line'}, default='normal'
            Selects the result quantity to display.
        bar_mesh_type : {'bars', 'user_mesh', 'mesh'}, default='bars'
            Mesh used for the graphic bar geometry.
        result_mesh_type : {'bars', 'user_mesh', 'mesh'}, default='mesh'
            Mesh used for plotting the result distribution.
        decimals : int, optional
            Number of decimals for label annotation.
        n_disc : int, default=10
            Number of subdivisions for result interpolation.

        Raises
        ------
        ValueError
            If the mode is invalid.
        """
        if mode not in ['real', 'virt']:
            msg = (f'Mode has to be either "real" or "virt".'
                   f'Got {mode} instead.')
            self.logger.error(msg)
            raise ValueError(msg)

        if mode == 'real':
            return self.solution_real_system.plot(
                kind, bar_mesh_type, result_mesh_type, decimals, n_disc)
        else:
            return self.solution_virtual_system.plot(
                kind, bar_mesh_type, result_mesh_type, decimals, n_disc)

    def _validate_virtual_load(self, force: Literal['fx', 'fz', 'fm']):
        """Validates the specified virtual load component.

        Parameters
        ----------
        force : {'fx', 'fz', 'fm'}
            Component of the virtual load (axial, shear, or moment).

        Raises
        ------
        ValueError
            If the load component is not valid.
        """
        self.logger.info(
            "Starting the validation of the virtual load components.")
        if force not in ['fx', 'fz', 'fm']:
            msg = f"Force component must be 'fx', 'fz' or 'fm'. Got {force}."
            self.logger.error(msg)
            raise ValueError(msg)

    def _validate_virtual_system(self):
        """Ensures that a virtual system has been created.

        Required before computing a deformation or accessing solution objects.

        Raises
        ------
        RuntimeError
            If no virtual load has been applied and therefore no virtual
            system exists.
        """
        self.logger.info(
            "Starting the validation of the virtual system.")
        if self.__virt_system is None:
            msg = ("No virtual system has been created yet. "
                   "Apply a virtual node or bar load first.")
            self.logger.error(msg)
            raise RuntimeError(msg)


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
            Wenn der Lagertyp nicht erkannt wird.
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
            Wenn der Gelenktyp nicht erkannt wird.
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


@dataclass(eq=False)
class ForceMethod(ReductionTheorem):
    """Executes the Force Method

    Method to calculate internal forces, support forces and deformations of
    statically undetermined systems (n > 0). This method is based on the
    principle of virtual forces (PvF).

    Parameters
    ----------
    system : :any:`System`
        The structural system to be analyzed using the Force Method.
    """

    def __post_init__(self):
        """Initialize reduction workflow and state.

        Calls the base initializer and prepares an internal snapshot of
        the modifier after each release operation to track the reduced
        configuration.
        """
        self._system_modifier = SystemModifier(system=self.system)
        self._released_modifier = None
        self.logger.info("Force Method has successfully been created.")

    @cached_property
    def uls_systems(self):
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
        self.logger.info(
            "Accessing method to generate ULS (unit load state) systems.")
        self._validate_released_system()

        self._system_modifier.delete_loads()
        self.logger.debug(
            "All existing loads have been removed from the released system.")
        uls = self._system_modifier.create_uls_systems()
        self.logger.debug(
            f"{len(uls)} ULS systems have been successfully generated.")
        return uls

    @cached_property
    def solution_uls_systems(self):
        r"""Returns the first order analysis of the unit load states.

        A new :any:`FirstOrder` instance is created for each unit load system.
        Before solving, the system is meshed consistently with the real load
        state system, ensuring matching integration lengths for the internal
        work calculation.

        Returns
        -------
        list[:any:`FirstOrder`]
            List of generated FirstOrder instances, one per unit load case.
        """
        self.logger.info("Creating FirstOrder instances for the ULS systems.")
        solution = []
        for i, uls_system in enumerate(self.uls_systems):
            uls_system.create_mesh(
                self._system_modifier.division_positions_for(uls_system))
            self.logger.debug(
                f"ULS system {i} has been meshed for solution computation.")
            solution.append(FirstOrder(uls_system, debug=self.debug))
            self.logger.debug(
                f"FirstOrder instance for ULS system {i} created "
                f"successfully.")
        return solution

    @cached_property
    def solution_rls_system(self):
        r"""Returns the first order analysis of the released system in the
        real load state.

        A new :any:`FirstOrder` instance is created using the released
        system with real loads.

        Returns
        -------
        :any:`FirstOrder`
            FirstOrder instance operating on the released system with real
            loads.
        """
        self.logger.info(
            "Creating FirstOrder instances for the released system with real "
            "loads (real load state).")
        self.released_system.create_mesh(
            self._system_modifier.division_positions_mesh())
        self.logger.debug(
            "RLS system has been meshed for solution computation.")
        solution = FirstOrder(self.released_system, debug=self.debug)
        self.logger.debug(
            "FirstOrder instance for RLS system created successfully.")
        return solution

    @property
    def eow_vector(self):
        return [
            EquationOfWork(self.solution_rls_system, uls)
            for uls in self.solution_uls_systems
        ]

    @cached_property
    def eow_matrix(self):
        return np.array(
            [[EquationOfWork(uls_i, uls_j)
              for uls_j in self.solution_uls_systems]
             for uls_i in self.solution_uls_systems],
            dtype=object
        )

    @cached_property
    def load_coef(self):
        """
        Calculate the load coefficients for the force method.

        Evaluates the Equation of Work between the released real load
        system (RLS) and each unit load system (ULS). The resulting
        coefficients form the right-hand side vector of the force method
        equation system.

        Returns
        -------
        :any:`numpy.array`
            One-dimensional array of preliminary coefficients, with one
            entry per redundant.
        """
        return np.array(
            [[eow.delta_ij] for eow in self.eow_vector],
            dtype=float
        )

    def load_coef_test(self):
        return np.array(
            [[eow.delta_ij] for eow in self.eow_vector],
            dtype=float
        )

    @cached_property
    def influence_coef(self):
        """Calculate the influence number matrix for the force method.

        Evaluates the Equation of Work between all combinations of unit
        load systems (ULS). The resulting square matrix contains the
        coefficients that represent the coupling between redundants.
        This matrix is used as the system matrix in the force method.

        Returns
        -------
        :any:`numpy.array`
            Two-dimensional square matrix of influence numbers, with one
            row and one column for each redundant.
        """
        self.logger.info("Accessing the calculation of the influence number "
                         "matrix.")
        matrix = np.array(
            [[eow.delta_ij for eow in row]
             for row in self.eow_matrix],
            dtype=float)
        self.logger.debug("The ")
        return matrix

    @property
    def redundants(self):
        """
        Solve the linear system to compute the redundant forces.

        Forms the system of equations A x = b, where A is the matrix of
        influence numbers (from :meth:`load_coef`) and b is the vector of
        preliminary coefficients (from :meth:`preliminary_coef`). Solving
        this system yields the values of the redundant forces.

        Returns
        -------
        :any:`numpy.array`
            One-dimensional array containing the solved redundant forces.
        """
        return np.linalg.solve(self.influence_coef, -self.load_coef)

    def plot(
            self, mode: Literal['uls', 'rls'] = 'rls',
            uls_index: int | None = None,
            kind: Literal[
                'normal', 'shear', 'moment', 'u', 'w', 'phi',
                'bending_line'] = 'normal',
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            result_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: int | None = None, n_disc: int = 10
    ):
        r"""Plot internal forces or deformation results of either the system
        with real loads, the unit load states or the real load state.

        Parameters
        ----------
        mode : {'uls', 'rls'}, default='rls'
            Defines whether the results of a unit load state or the real load
            state.
        uls_index : :any:`int`| None
            If the chosen mode is uls, then an index of the unit loads systems
            is needed to plot the chosen system.
        kind : {'normal', 'shear', 'moment', 'u', 'w', 'phi', 'bending_line'},
                default='normal'
            Selects the result quantity to display.
        bar_mesh_type : {'bars', 'user_mesh', 'mesh'}, default='bars'
            Mesh used for the graphic bar geometry.
        result_mesh_type : {'bars', 'user_mesh', 'mesh'}, default='mesh'
            Mesh used for plotting the result distribution.
        decimals : int, optional
            Number of decimals for label annotation.
        n_disc : int, default=10
            Number of subdivisions for result interpolation.

        Raises
        ------
        ValueError
            If the mode is invalid.
        """
        if mode not in ['uls', 'rls']:
            raise ValueError(
                f'Mode has to be either "uls" or "rls".'
                f'Got {mode} instead.'
            )
        if mode in ['rls'] and uls_index is not None:
            raise ValueError(
                f'If the mode is set to "rls", the '
                f'uls_index has to be set to None. Got mode: {mode} and'
                f'uls_index: {uls_index} instead.'
            )
        if mode == 'uls' and uls_index is None:
            raise ValueError(
                f'If the mode is set to "uls", the index can not be None.'
                f'Got mode: {mode} and index: {uls_index} instead.'
            )
        if mode == 'rls':
            return self.solution_rls_system.plot(
                kind, bar_mesh_type, result_mesh_type, decimals, n_disc)
        else:
            return self.solution_uls_systems[uls_index].plot(
                kind, bar_mesh_type, result_mesh_type, decimals, n_disc)

    def _validate_virtual_load(self, force):
        raise ValueError("Virtual loads are not allowed for this method.")

    def _validate_virtual_system(self):
        raise ValueError(
            "There is not a virtual system being defined in this method.")

    def add_virtual_node_load(self, *args, **kwargs):
        """Disallows virtual node loads for KGV.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        raise ValueError(
            "Virtual node loads are not allowed for this method.")

    def add_virtual_bar_load(self, *args, **kwargs):
        """Disallows virtual bar loads for KGV.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        raise ValueError(
            "Virtual bar loads are not allowed for this method.")

    def add_virtual_moment_couple(self, *args, **kwargs):
        """Disallows virtual moment couple for KGV.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        raise ValueError(
            "Virtual moment couple is not allowed for this method.")

    def virtual_system(self):
        raise ValueError("There is not a virtual system in this method.")

    def solution_virtual_system(self):
        raise ValueError("There is not a virtual solution in this method.")

    def deformation(self):
        """Disallows computation of single deformations in KGV.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        raise ValueError("Deformation cannot be computed for this method.")


class DMG(ForceMethod):

    support_moment = ForceMethod.redundants

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
        A = self.influence_coef * 6
        b = self.load_coef.reshape(-1, 1) * 6
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
