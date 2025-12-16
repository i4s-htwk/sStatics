
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from typing import Literal

from sstatics.core.logger_mixin import LoggerMixin
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.postprocessing.equation_of_work import EquationOfWork
from sstatics.core.calc_methods import FirstOrder


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

    def _work_of_bar(self, matrix, bar, sum: bool = True):
        """
        Return the work row(s) associated with a Bar object.

        If the Bar corresponds to a single mesh segment, its work row is
        returned directly. If the Bar is composed of multiple mesh segments,
        the corresponding rows are stacked. If ``sum`` is True, all rows are
        summed into a single vector.

        Parameters
        ----------
        matrix : ndarray
            Work matrix for bars (rows correspond to mesh segments).
        bar : Bar
            The Bar object whose work contribution must be requested.
        sum : bool, optional
            If True (default), all mesh segment rows are summed into a single
            vector. If False, all rows are returned as a matrix.

        Returns
        -------
        ndarray
            A 1D array for a single row or summed rows, or a 2D array for all
            rows if ``sum`` is False.

        Raises
        ------
        ValueError
            If the Bar has no mesh segments or a segment is missing from
            ``system.mesh``.
        """
        mesh = self.system.mesh

        # --- Case 1: single mesh segment ------------------------------------
        try:
            idx = mesh.index(bar)
            self.logger.debug(
                f"Bar {bar} is a single mesh segment (index {idx}).")
            return matrix[idx]
        except ValueError:
            self.logger.debug(
                f"Bar {bar} is not a single mesh segment "
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

            rows.append(matrix[idx][:5])
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

    def _work_of_node(self, matrix, node):
        """
        Return the work row associated with a Node object.

        Retrieves the row from the nodal work matrix corresponding to the
        specified node.

        Parameters
        ----------
        matrix : ndarray
            Work matrix for nodes (rows correspond to nodes).
        node : Node
            The Node object whose work contribution is requested.

        Returns
        -------
        ndarray
            The work row associated with the given Node.

        Raises
        ------
        ValueError
            If the Node is not contained in the system.
        """
        nodes = self.system.nodes()

        try:
            idx = nodes.index(node)
        except ValueError:
            msg = f"Node {node} not found in system."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(f"Node {node} → index {idx}")
        return matrix[idx]

    def work_of(self, obj: Node | Bar, sum: bool = True):
        """
        Return the work contribution for a Node or a Bar.

        For a Bar:
            Returns one or multiple work rows depending on the number of
            mesh segments belonging to the Bar. If ``sum`` is True, the rows
            are summed into a single vector.

        For a Node:
            Returns the single work row associated with that Node.

        Parameters
        ----------
        obj : Node | Bar
            The structural object whose work contribution is requested.
        sum : bool, optional
            Summation flag for Bars with multiple mesh segments. Ignored
            for Nodes. Default is True.

        Returns
        -------
        ndarray
            A 1D array representing the work row (Node) or summed row(s) (Bar),
            or a 2D array of rows if ``sum`` is False for a Bar.

        Raises
        ------
        TypeError
            If ``obj`` is neither a Node nor a Bar.
        """
        self.logger.debug(f"Accessing work_of for {obj} with sum={sum}")
        if isinstance(obj, Bar):
            result = self._work_of_bar(
                matrix=self.work_matrix('bars'),
                bar=obj,
                sum=sum
            )
            self.logger.info(
                f"Returning work for Bar {obj}.")
            return result

        if isinstance(obj, Node):
            result = self._work_of_node(
                matrix=self.work_matrix('nodes'),
                node=obj
            )
            self.logger.info(
                f"Returning work for Node {obj}.")
            return result
        msg = f"Expected Node or Bar, got {type(obj).__name__}"
        self.logger.error(msg)
        raise TypeError(msg)

    def work_matrix(self, kind: Literal['nodes', 'bars']):
        """
        Return the work matrix for nodes or bars.

        Selects the internal work matrix corresponding to either Nodes or
        Bars. Rows correspond to the degrees of freedom for nodes or mesh
        segments for bars.

        Parameters
        ----------
        kind : {'nodes', 'bars'}
            Specify whether to return the nodal work matrix or the bar work
            matrix.

        Returns
        -------
        ndarray
            The requested work matrix.

        Raises
        ------
        ValueError
            If ``kind`` is not 'nodes' or 'bars'.
        """
        self.logger.debug(f"Accessing work_matrix for kind='{kind}'")
        if kind == 'bars':
            self.logger.info("Returning work matrix for Bars")
            return self.equation_of_work.work_matrix_bars
        if kind == 'nodes':
            self.logger.info("Returning work matrix for Nodes")
            return self.equation_of_work.work_matrix_nodes
        msg = f"kind must be 'nodes' or 'bars', got {kind}"
        self.logger.error(msg)
        raise ValueError(msg)

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
