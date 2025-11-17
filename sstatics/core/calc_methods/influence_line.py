
from dataclasses import dataclass, field
from typing import Literal, List

import numpy as np

from sstatics.core.logger_mixin import LoggerMixin
from sstatics.core.preprocessing import Bar, Node, SystemModifier, System
from sstatics.core.solution import Solver
from sstatics.core.solution.poleplan.operation import get_angle
from sstatics.core.postprocessing import BarResult, RigidBodyDisplacement


# TODO: FirstOrder oder Solver? -> Plot
@dataclass(eq=False)
class InfluenceLine(LoggerMixin):
    """
    Computes influence lines of internal forces and displacements using the
    kinematic method.

    The class provides two main analysis modes: :meth:`force` for internal
    forces and :meth:`deform` for displacements. Both methods determine the
    influence line of a specified quantity at a given location within the
    structure.

    Notes
    -----
    An influence line is a function that describes how a moving unit load
    affects a force or displacement quantity at a fixed location within the
    structure [1]_.

    References
    ----------
    .. [1] D. Dinkler. "Grundlagen der Baustatik: Modelle und
           Berechnungsmethoden für ebene Stabtragwerke". Band 1, Seite 128,
           2011.

    See Also
    --------
    :py:class:`SystemModifier` :
        Class for modifying systems to compute influence lines.
    :py:class:`FirstOrder` :
        Class for solving first-order systems.
    :py:class:`BarResult` :
        Class for storing bar results.
    """

    system: System
    debug: bool = False

    _modified_system: System | None = field(init=False, default=None)
    _solution: Solver | None = field(init=False, default=None)
    _norm_force: float | None = field(init=False, default=None)
    _deflections: List[BarResult] | None = field(init=False, default=None)
    _poleplan: None = field(init=False, default=None)
    _rigid_motions: List[RigidBodyDisplacement] | None = (
        field(init=False, default=None))

    def force(
            self,
            kind: Literal['fx', 'fz', 'fm'],
            obj: Bar | Node,
            position: float = 0.0,
            n_disc: int = 10
    ):
        r"""
        Computes the influence line for the specified force quantity at the
        given location and then calls 'plot()' to display the graphical
        result.

        Parameters
        ----------
        kind : Literal['fx', 'fz', 'fm']
            Type of the target force quantity.
        obj : Bar | Node
            Location — Node or Bar.
        position : float, default=0.0
            If obj is a Bar, the position along the member.
        n_disc : int, default=10
            Number of discretization points for plotting.

        Returns
        -------
        None
            Plots the influence line.

        Notes
        -----
        The influence line is determined using the kinematic method.

        1. Lagrange's release - Release the constraint conjugate to the desired
           force quantity at location k (position) and apply a unit load to
           account for the imposed negative displacement [1]_.
        2. Element stiffness matrix - Formulate stiffness relations of
           individual elements considering hinges [2]_.
        3. Global system assembly - Establish equation system for entire
           structure including support conditions [2]_.
        4. Mobility check - Use singular value decomposition to check for
           mobility [1]_.
        5. Solve equation system - Determine unknown nodal displacements, then
           compute total deformation at bar ends.
        6. Scale results - Multiply member deformations and end forces by
           :math:`f = -1 / \delta_{k,act}` [3]_.
        7. Deflection curve - Determine deflection curve using beam
           differential equation.

        References
        ----------
        .. [1] D. Dinkler. "Grundlagen der Baustatik: Modelle und
               Berechnungsmethoden für ebene Stabtragwerke". Band 1,
               Seite 105 ff., 2011.

        .. [2] R. Dallmann. "Baustatik 3: Theorie II. Ordnung und
               computerorientierte Methoden der Stabtragwerke". Band 2,
               Seite 65 ff., 2015.

        .. [3] W. Franke, T. Kunow. "Kleines Einmaleins der Baustatik", 2007.

        See Also
        --------
        :py:meth:`_compute_norm_force` :
            Method to compute the normalization force.
        :py:meth:`_modify_system` :
            Method to modify the system for influence calculation.
        :py:class:`SystemModifier` :
            Class for system modifications.
        :py:class:`BarResult` :
            Class for bar results.
        """
        self._reset_results()

        if kind not in ('fx', 'fz', 'fm'):
            raise ValueError(f"Invalid kind type: {kind}")

        self._modified_system = self._modify_system(kind, obj, position)
        self._solution = Solver(system=self._modified_system)

        if self.solution.solvable:
            self._norm_force = self._compute_norm_force(kind, obj)
            if self._norm_force != 0:
                self._modified_system = self._modify_system(
                    kind, obj, position, virt_force=self.norm_force
                )
                self._solution = Solver(system=self._modified_system)
            self._create_deflection_objects(n_disc=n_disc)

        else:
            from sstatics.core.solution import Poleplan

            self._poleplan = Poleplan(system=self.modified_system,
                                      debug=self.debug)
            chain, angle = self._compute_chain_angle(kind, obj, position)
            self._poleplan.set_angle(target_chain=chain, target_angle=angle)
            self._rigid_motions = self._poleplan.rigid_motion(n_disc=n_disc)

        self.plot()

    def deform(
            self,
            kind: Literal['u', 'w', 'phi'],
            obj,
            position: float = 0,
            n_disc: int = 10
    ):
        """
        Computes the influence line for the specified displacement quantity
        at the given location and then calls 'plot()' to display the
        graphical result.

        Parameters
        ----------
        kind : Literal['u', 'w', 'phi']
            Type of the target displacement quantity.
        obj : Bar | Node
            Location — Node or Bar.
        position : float, default=0
            If obj is a Bar, the position along the member.
        n_disc : int, default=10
            Number of discretization points for plotting.

        Returns
        -------
        None
            Plots the influence line.

        Notes
        -----
        The theory underlying this algorithm can be found in [1]_.

        1. Apply unit load conjugate to desired displacement quantity.
        2. Split loaded bar into two sub-bars and assign load to new node.
        3. Solve for unknown nodal displacements, derive deformations and
           end forces.
        4. Determine deflection curve using beam differential equation.

        References
        ----------
        .. [1] D. Dinkler. "Grundlagen der Baustatik: Modelle und
               Berechnungsmethoden für ebene Stabtragwerke". Band 1, 2011.

        See Also
        --------
        :py:meth:`_create_deflection_objects` :
            Method to create deflection objects.
        :py:class:`BarResult` :
            Class for bar results.
        """
        self._reset_results()

        if kind not in ['u', 'w', 'phi']:
            raise ValueError(f"Invalid kind type: {kind}")

        self._modified_system = self._modify_system(kind, obj, position)
        self._solution = Solver(system=self._modified_system)

        self._create_deflection_objects(n_disc=n_disc)

        self.plot()

    @property
    def modified_system(self) -> System:
        """
        The modified system used for the influence line computation.

        Returns
        -------
        System
        """
        if self._modified_system is None:
            raise AttributeError(
                "The modified system has not been created yet. "
                "Call `force()` or `deform()` before accessing "
                "`modified_system`."
            )
        return self._modified_system

    @property
    def solution(self) -> Solver:
        """
        The solution of the system used for the influence line computation.

        Returns
        -------
        Solver
        """
        if self._solution is None:
            raise AttributeError(
                "No solution is available. "
                "Call `force()` or `deform()` before accessing `solution`."
            )
        return self._solution

    @property
    def norm_force(self) -> float:
        """
        The normalizing virtual force used for the influence line computation.

        Returns
        -------
        float
        """
        if self._norm_force is None:
            raise AttributeError(
                "The normalizing virtual force is not defined. "
                "Call `force()` before accessing `norm_force`."
            )
        return self._norm_force

    @property
    def poleplan(self):
        """
        The pole plan used for the influence line computation.

        Returns
        -------
        Poleplan
        """
        if self._poleplan is None:
            if self._solution is not None:
                raise ValueError(
                    "No pole plan is available because the influence line "
                    "was solved by the deformation method. "
                    "A pole plan is only used when the system becomes a "
                    "mechanism."
                )
            raise AttributeError(
                "No pole plan has been computed yet. "
                "Call `force()` before accessing `poleplan`."
            )
        return self._poleplan

    @property
    def deflections(self) -> List[BarResult]:
        """
        The deflection curves of the members in the analysis mesh.

        Returns
        -------
        List[BarResult]
        """
        if self._solution is None:
            raise AttributeError(
                "No deflection data available. "
                "Call `force()` or `deform()` before accessing `deflections`."
            )
        if self._deflections is None and not self.solution.solvable:
            raise AttributeError(
                "No deflection data available because the system became "
                "a mechanism. The rigid-body displacement can be accessed via "
                "`rigid_motions`."
            )
        return self._deflections

    @property
    def rigid_motions(self) -> List[RigidBodyDisplacement]:
        """
        The rigid-body motions of the members when the system becomes a
        mechanism.

        Returns
        -------
        List[RigidBodyDisplacement]
        """
        if self._rigid_motions is None:
            raise AttributeError(
                "No rigid-body motion data available. "
                "Call `force()` before accessing `rigid_motions`. "
                "Rigid-body motions are only generated when the system "
                "becomes a mechanism."
            )
        return self._rigid_motions

    def _modify_system(
            self,
            kind: Literal['fx', 'fz', 'fm', 'u', 'w', 'phi'],
            obj: Bar | Node,
            position: float,
            virt_force: float = 1) -> System:
        """
        Internal method for invoking the appropriate `SystemModifier`
        methods depending on whether `kind` refers to a force or a
        displacement quantity.

        Parameters
        ----------
        kind : Literal['fx', 'fz', 'fm', 'u', 'w', 'phi']
            Type of the desired force or displacement quantity.
        obj : Bar | Node
            Location — node (Node) or member (Bar).
        position : float
            If obj is a Bar, the position along the member.
        virt_force : float, default=1
            Virtual force to be applied.

        Returns
        -------
        System
            The modified system.

        Raises
        ------
        TypeError
            If obj is not an instance of Bar or Node.
        ValueError
            If obj is a Node and position is not 0.
        ValueError
            If kind is invalid.

        See Also
        --------
        :py:class:`SystemModifier` :
            Class for modifying systems.
        """
        modifier = SystemModifier(self.system)
        is_bar = isinstance(obj, Bar)
        is_node = isinstance(obj, Node)

        if not (is_bar or is_node):
            raise TypeError("obj must be an instance of Bar or Node.")

        if is_node and position != 0:
            raise ValueError("If obj is a Node, `position` must be 0.")

        force_kinds = {'fx', 'fz', 'fm'}
        deform_kinds = {'u', 'w', 'phi'}

        if kind in force_kinds:
            if is_bar:
                return modifier.modify_bar_force_influ(
                    obj, kind, position, virt_force
                )
            return modifier.modify_node_force_influ(obj, kind, virt_force)

        elif kind in deform_kinds:
            if is_bar:
                return modifier.modify_bar_deform_influ(obj, kind, position)
            return modifier.modify_node_deform(obj, kind, position)

        else:
            force_deform_kinds = force_kinds | deform_kinds
            raise ValueError(
                f"Invalid kind '{kind}'. Must be one of {force_deform_kinds}.")

    def _compute_norm_force(self, force: Literal['fx', 'fz', 'fm'],
                            obj: Bar | Node) -> float:
        """
        Determines the scaling factor f = −1/δ from the computed
        displacement δ (from the displacement method) at the location
        (position) of the desired force influence line.

        Parameters
        ----------
        force : Literal['fx', 'fz', 'fm']
            Type of the force quantity.
        obj : Bar | Node
            Location where the influence line of the force quantity is
            sought.

        Returns
        -------
        float
            The normalization force.

        Raises
        ------
        TypeError
            If obj is not an instance of Bar or Node.
        ZeroDivisionError
            If delta is zero.

        See Also
        --------
        :py:meth:`force` : Method that uses this normalization factor.
        """
        delta = 0.0
        if isinstance(obj, Bar):
            deform = self.solution.bar_deform_list
            bars = list(self.system.bars)
            idx = bars.index(obj)

            d_i, d_j = deform[idx], deform[idx + 1]
            mapping = {'fx': (3, 0), 'fz': (4, 1), 'fm': (5, 2)}
            i, j = mapping[force]

            delta = d_j[j][0] - d_i[i][0]
        elif isinstance(obj, Node):
            node_deform = self.solution.node_deform
            for i, n in enumerate(self.system.nodes()):
                if n is obj:
                    slice_ = slice(i * 3, i * 3 + 3)
                    nd = node_deform[slice_, :]
                    mapping = {'fx': 0, 'fz': 1, 'fm': 2}
                    delta = nd[mapping[force]][0]
                    break
        else:
            raise TypeError("obj must be an instance of Bar or Node")
        if delta == 0:
            raise ZeroDivisionError(
                "Delta is zero – cannot compute norm force.")
        return -float(np.abs(1 / delta))

    def _compute_chain_angle(self, force: Literal['fx', 'fz', 'fm'],
                             obj: Bar | Node,
                             position: float):
        """
        Determines the rotation angle to be applied to the kinematic chain
        in the pole plan such that the virtual displacement corresponding
        to the desired force influence line equals 1 at the specified
        location.

        Parameters
        ----------
        force : Literal['fx', 'fz', 'fm']
            Type of the force quantity.
        obj : Bar | Node
            Location of the desired force quantity.
        position : float
            If obj is a Bar, the position along the member.

        Returns
        -------
        tuple
            A tuple containing the chain and the angle.

        Raises
        ------
        AttributeError
            If no valid chain is found.

        See Also
        --------
        :py:meth:`force` : Method that uses this angle calculation.
        """
        angle = 0
        if isinstance(obj, Bar):
            idx = list(self.system.bars).index(obj)
            bar = self.modified_system.bars[idx]
            chain = self._poleplan.get_chain(bars={bar})

            if chain is None:
                raise AttributeError("No valid chain found for the given bar.")

            if force == 'fz':
                if position in {0, 1}:
                    node = obj.node_i if position == 0 else obj.node_j
                    displacement = 1 if position == 0 else -1

                    if chain.absolute_pole.is_infinite:
                        aPole_coords, node_coords, c = (
                            self._poleplan.find_adjacent_chain(node, chain)
                        )
                        if aPole_coords is None:
                            for rPole in chain.relative_pole:
                                if rPole != node:
                                    aPole_coords, node_coords, c = (
                                        self._poleplan.find_adjacent_chain(
                                            rPole.node, chain)
                                    )
                        idx_chain = self._poleplan.chains.index(chain)
                        next_chain = self._poleplan.chains.index(c)

                        angle = get_angle(point=node_coords,
                                          center=aPole_coords,
                                          displacement=displacement)
                        if idx_chain < next_chain:
                            angle = angle / c.angle_factor
                    else:
                        aPole_coords = chain.absolute_pole.coords
                        node_coords = np.array([[node.x], [node.z]])

                        angle = get_angle(point=node_coords,
                                          center=aPole_coords,
                                          displacement=displacement)
                else:
                    angle = -1 / obj.length
            elif force == 'fm':
                if position == 0:
                    angle = -1
                elif position == 1:
                    angle = 1
                else:
                    angle = (1 - position) / obj.length

            return chain, angle
        elif isinstance(obj, Node):
            chain = self._poleplan.get_chain_node(obj)
            if chain is None:
                raise AttributeError(
                    "No valid chain found for the given node."
                )

            if force == 'fz':
                aPole_coords = chain.absolute_pole.coords
                node_coords = np.array([[obj.x], [obj.z]])

                angle = get_angle(point=node_coords,
                                  center=aPole_coords,
                                  displacement=1)
            elif force == 'fm':
                angle = -1

            return chain, angle

    def _create_deflection_objects(self, n_disc: int = 10):
        """
        Determines the differential equations of the deflection curves from
        the results of the displacement method (`solution`) using the class
        'BarResult'. The deflection equations of all members in the analysis
        mesh are stored in the private attribute `self._deflections`.

        Parameters
        ----------
        n_disc : int, default=10
            Number of discretization points for plotting.

        Returns
        -------
        None

        See Also
        --------
        :py:class:`BarResult` : Class for storing bar results.
        """
        deflections = []
        for i, bar in enumerate(self.modified_system.mesh):
            dgl = BarResult(
                bar=bar,
                forces=self.solution.internal_forces[i],
                deform=self.solution.bar_deform_list[i],
                n_disc=n_disc
            )
            deflections.append(dgl)
        self._deflections = deflections

    def _reset_results(self):
        """
        Resets all computed attributes to None before a new calculation.

        Returns
        -------
        None
        """
        self._modified_system = None
        self._solution = None
        self._norm_force = None
        self._deflections = None
        self._poleplan = None
        self._rigid_motions = None

    def plot(self, mode: str = 'MPL'):
        """
        Plot the influence line, either based on deformation results or
        rigid-body motion results.

        Parameters
        ----------
        mode : str, default='MPL'
            Rendering mode for graphical output.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If no influence line data is available.
        """
        if self._deflections is not None:
            from sstatics.graphic_objects import ResultGraphic
            from sstatics.core.postprocessing import SystemResult

            sol = self.solution
            result = SystemResult(
                system=self.modified_system,
                bar_deform_list=sol.bar_deform_list,
                bar_internal_forces=sol.internal_forces,
                node_deform=sol.node_deform,
                node_support_forces=sol.node_support_forces,
                system_support_forces=sol.system_support_forces,
            )
            result.bars = self.deflections
            ResultGraphic(system_result=result, kind='w').show()
        elif self._rigid_motions is not None:
            from sstatics.graphic_objects.poleplan import PoleplanGraphic
            PoleplanGraphic(poleplan=self.poleplan).show()
        else:
            print(mode)
            raise AttributeError(
                "No influence line data found. "
                "Call `force()` or `deform()` before using `plot()`."
            )
