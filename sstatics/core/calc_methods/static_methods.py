
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
    _equation_of_work: EquationOfWork = field(init=False, default=None)

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

    def _create_equation_of_work(self):
        """
        Create the ``EquationOfWork`` instance safely.

        This method instantiates ``EquationOfWork`` once and provides full
        error handling. It logs the creation attempt and reports detailed
        error messages if invalid input, runtime issues, or unexpected
        exceptions occur.

        Returns
        -------
        EquationOfWork
            The created ``EquationOfWork`` instance.

        Raises
        ------
        ValueError
            If the provided solution data is invalid.
        RuntimeError
            If a runtime or unexpected error occurs during creation.
        """
        try:
            self.logger.info("Creating EquationOfWork...")
            eq = EquationOfWork(
                solution_i=self.solution_real_system,
                solution_j=self.solution_virtual_system,
                debug=self.debug
            )
            self.logger.info("EquationOfWork successfully created.")
            return eq

        except ValueError as e:
            msg = f"Invalid data while creating EquationOfWork: {e}"
            self.logger.error(msg)
            raise ValueError(msg) from e

        except RuntimeError as e:
            msg = f"Runtime error during EquationOfWork creation: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        except Exception as e:
            msg = f"Unexpected error in EquationOfWork: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

    @property
    def equation_of_work(self):
        """
        EquationOfWork

        Returns the cached ``EquationOfWork`` instance. If it has not been
        created yet, it is instantiated on first access.

        Returns
        -------
        EquationOfWork
            The lazily created and cached ``EquationOfWork`` instance.
        """
        if self._equation_of_work is None:
            self._equation_of_work = self._create_equation_of_work()
        return self._equation_of_work

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

        deformation = self.equation_of_work.delta_ij

        self.logger.debug(f"Computed deformation value: {deformation}")
        return deformation

    @property
    def work_matrix_nodes(self):
        """
        ndarray

        Returns the work matrix for all nodes. The matrix is obtained from
        the lazily created ``EquationOfWork`` instance. Each row contains
        the work contributions associated with a specific node.

        Returns
        -------
        ndarray
            Matrix of nodal work values.
        """
        return self.equation_of_work.work_matrix_nodes

    @property
    def work_matrix_bars(self):
        """
        ndarray

        Returns the work matrix for all bars. The matrix is provided by the
        ``EquationOfWork`` instance and contains work contributions for each
        bar or mesh segment.

        Returns
        -------
        ndarray
            Matrix of bar work values.
        """
        return self.equation_of_work.work_matrix_bars

    def work_of_bar(self, bar, sum: bool = True):
        """
        Return the work row(s) associated with a bar.

        If the bar corresponds directly to a single mesh segment, its work
        row is returned. If the bar consists of multiple mesh segments, the
        corresponding rows are stacked. When ``sum`` is True, the stacked
        rows are summed to produce a single work vector.

        Parameters
        ----------
        bar : Bar
            The bar for which the work values are requested.
        sum : bool, optional
            If True, the segment rows are summed into a single vector.
            If False, all rows are returned. Default is True.

        Returns
        -------
        ndarray
            Either a single work row or a matrix of stacked rows.

        Raises
        ------
        ValueError
            If the bar has no associated mesh segments or a segment is
            missing in ``system.mesh``.
        """

        mesh = self.system.mesh
        wm = self.work_matrix_bars

        # --- Case 1: single mesh segment ------------------------------------
        try:
            idx = mesh.index(bar)
            self.logger.debug(
                f"Bar {bar} is a single mesh segment (index {idx}).")
            return wm[idx]
        except ValueError:
            self.logger.debug(
                f"[work_of_bar] Bar {bar} is not a single mesh segment "
                "→ checking composed segments.")

        # --- Case 2: composed of multiple mesh segments ---------------------
        segments = self.system.mesh_segments_of(bar)

        if not segments:
            msg = f"Bar {bar} has no mesh segments associated."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(
            f"Bar {bar} consists of {len(segments)} "
            f"mesh segments: {segments}"
        )

        # Collect rows (first 5 entries each)
        rows = []
        for seg in segments:
            try:
                idx = mesh.index(seg)
            except ValueError:
                self.logger.error(
                    f"Segment {seg} not found in system.mesh.")
                raise

            rows.append(wm[idx][:5])
            self.logger.debug(
                f"Segment {seg} → row index {idx}")

        rows = np.vstack(rows)

        if sum:
            self.logger.debug(
                f"Summing {len(segments)} rows for "
                f"bar {bar}: result = {rows.sum(axis=0)}"
            )
            return rows.sum(axis=0)

        self.logger.debug(
            "Returning full row matrix for the given bar "
            f"with shape {rows.shape}"
        )
        return rows

    def work_of_node(self, node):
        """
        Return the work row for a node.

        The method retrieves the row from the nodal work matrix associated
        with the given node. A descriptive error is raised if the node is
        not part of the system.

        Parameters
        ----------
        node : Node
            The node whose work row is requested.

        Returns
        -------
        ndarray
            The work row belonging to the given node.

        Raises
        ------
        ValueError
            If the node is not contained in the system.
        """
        nodes = self.system.nodes()

        try:
            idx = nodes.index(node)
        except ValueError:
            msg = f"Node {node} not found in system."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(f"Node {node} → index {idx}")
        return self.work_matrix_nodes[idx]

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
        r"""Returns the vector of Equation of Work objects for the real load
        state.

        For each unit load system (ULS), an :any:`EquationOfWork` instance is
        created, representing the internal work between the released real load
        system (RLS) and the respective ULS. This vector forms the basis for
        computing the load coefficients used in the force method.

        Returns
        -------
        :any:`numpy.array`
            nx1 vector containing one EquationOfWork instance per unit
            load system. n = number of unit load states
        """
        self.logger.info(
            "Creating EquationOfWork objects for the real load state "
            "(EOW vector).")
        vector = np.array(
            [[EquationOfWork(self.solution_rls_system, uls, debug=self.debug)]
             for uls in self.solution_uls_systems
             ], dtype=object)
        self.logger.debug(
            f"EOW vector created with {len(vector)} EquationOfWork instances.")
        return vector

    @cached_property
    def eow_matrix(self):
        r"""Returns the matrix of Equation of Work objects for all ULS
        combinations.

        Creates a square matrix where each entry represents the internal work
        between a pair of unit load systems (ULS). This matrix forms the basis
        for evaluating the influence coefficients used in the force method.

        Returns
        -------
        :any:`numpy.array`
            Two-dimensional object array containing EquationOfWork instances,
            with one row and one column per redundant.
        """
        self.logger.info(
            "Creating the EquationOfWork matrix for all ULS combinations.")
        matrix = np.array(
            [[EquationOfWork(uls_i, uls_j, debug=self.debug)
              for uls_j in self.solution_uls_systems]
             for uls_i in self.solution_uls_systems],
            dtype=object
        )
        self.logger.debug(
            f"EOW matrix created with shape {matrix.shape}.")
        return matrix

    @cached_property
    def load_coef(self):
        r"""Calculate the load coefficients for the force method.

        Evaluates the value :math:`\delta_{i}` from each EquationOfWork object
        of the real load state with a unit load state. The resulting vector
        forms the right-hand side of the linear system in the force method.

        Returns
        -------
        :any:`numpy.array`
            Column vector (n×1) of floating-point load coefficients.
        """
        self.logger.info("Calculating load coefficients (δᵢ vector).")
        coef = np.array(
            [[eow[0].delta_ij] for eow in self.eow_vector],
            dtype=float
        )
        self.logger.debug(
            f"Load coefficient vector created with shape {coef.shape}.")
        return coef

    @cached_property
    def influence_coef(self):
        r"""Calculate the influence number matrix for the force method.

        Converts the EquationOfWork matrix into its numerical form by
        evaluating :math:`\delta_{ij}` for each entry. The resulting square
        matrix represents the coupling between redundants and forms the
        stiffness-like system matrix of the force method.

        Returns
        -------
        :any:`numpy.array`
            Two-dimensional square matrix of floating-point influence numbers.
        """
        self.logger.info("Calculating influence number matrix (δᵢⱼ matrix).")
        matrix = np.array(
            [[eow.delta_ij for eow in row]
             for row in self.eow_matrix],
            dtype=float
        )
        self.logger.debug(
            f"Influence coefficient matrix created with shape {matrix.shape}.")
        return matrix

    @property
    def redundants(self):
        r"""Solve the system of equations for the redundant forces.

        Solves the linear system

        .. math::  A \, x = -b

        where

        * :math:`A` is the influence number matrix from
          :meth:`influence_coef`,
        * :math:`b` is the load coefficient vector from :meth:`load_coef`.

        The resulting vector :math:`x` contains the magnitudes of all redundant
        forces.

        Returns
        -------
        :any:`numpy.array`
            One-dimensional array containing the solved redundant forces.
        """
        self.logger.info("Solving linear system for redundant forces.")
        result = np.linalg.solve(self.influence_coef, -self.load_coef)
        self.logger.debug(
            f"Redundant forces computed: {result.flatten().tolist()}")
        return result

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
        uls_index : :any:`int` or None
            If the chosen mode is uls, then an index of the unit loads \
            system is needed to plot the chosen system.
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
        r"""Disallows virtual loads for this method.

        In the force method (KGV), virtual loads are not part of the
        computational procedure. Any attempt to define virtual loads is
        rejected.

        Raises
        ------
        ValueError
            Always raised, since virtual loads are incompatible with this
            method.
        """
        msg = "Virtual loads are not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def _validate_virtual_system(self):
        r"""Disallows virtual systems for this method.

        The force method does not make use of virtual systems. Any attempt to
        validate or access such a system is rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "There is not a virtual system being defined in this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def add_virtual_node_load(self, *args, **kwargs):
        r"""Disallows adding virtual node loads in the force method.

        Virtual node loads are a concept of the displacement method and are not
        used in the force method (KGV). Any attempt to add such loads is
        rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Virtual node loads are not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def add_virtual_bar_load(self, *args, **kwargs):
        r"""Disallows adding virtual bar loads in the force method.

        Since the force method does not employ virtual systems or virtual
        loads, defining bar loads in a virtual context is not permitted.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Virtual bar loads are not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def add_virtual_moment_couple(self, *args, **kwargs):
        r"""Disallows adding virtual moment couples in the force method.

        Virtual moment couples are incompatible with the solution strategy of
        the force method and are therefore rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Virtual moment couple is not allowed for this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def virtual_system(self):
        r"""Disallows access to a virtual system.

        The force method does not define or use a virtual system. Accessing one
        is not permitted.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "There is not a virtual system in this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def solution_virtual_system(self):
        r"""Disallows access to a virtual solution.

        Virtual solutions are not generated in the force method. Any attempt to
        retrieve such a solution is rejected.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "There is not a virtual solution in this method."
        self.logger.error(msg)
        raise ValueError(msg)

    def deformation(self):
        r"""Disallows computation of single deformations.

        In the force method (KGV), deformations are not computed directly by
        querying individual values. Deformations are only implicitly evaluated
        through the Equation of Work. Accessing single deformations is
        therefore not supported.

        Raises
        ------
        ValueError
            Always raised for this method.
        """
        msg = "Deformation cannot be computed for this method."
        self.logger.error(msg)
        raise ValueError(msg)


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
