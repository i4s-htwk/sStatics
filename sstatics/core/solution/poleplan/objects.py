
from dataclasses import dataclass, field
from typing import Optional, Type
import numpy as np

from sstatics.core.logger_mixin import LoggerMixin
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System


@dataclass(eq=False)
class Pole:
    """
    Represents a Pole in a Poleplan (Polplan) of a planar structural system.

    A Pole is a point in the plane of the structure around which a chain
    can rotate. Depending on its type and position, it defines the
    relative or absolute motion of connected elements.

    Types of Poles
    --------------
    - Absolute Pole (Hauptpol):
        A fixed point in the plane that does not translate. The associated
        body rotates around this point. Examples include fixed supports or
        rigid connections to fixed parts of the structure.

    - Relative Pole (Nebenpol):
        Defines the relative rotation between two connected bodies. For
        example, a moment hinge (M = 0) directly gives the relative pole.
        If shear or translational constraints are zero (N = 0 or Q = 0),
        the relative pole lies at infinity perpendicular to the motion
        direction. For non-adjacent bodies, the relative pole may lie
        somewhere in the plane.

    Attributes
    ----------
    node : Node
        The node associated with the Pole, providing coordinates in the
        plane.
    same_location : bool, default=False
        Indicates whether the Pole coincides exactly with the node
        coordinates.
    direction : float or None, default=None
        Direction of the Pole for translation or Poles at infinity (in
        radians). None if not applicable.
    is_infinite : bool, default=False
        Indicates whether the Pole is at infinity, corresponding to a
        translational motion.

    Properties
    ----------
    x : float or None
        X-coordinate of the Pole if `same_location=True`, else None.
    z : float or None
        Z-coordinate of the Pole if `same_location=True`, else None.
    coords : np.ndarray
        2x1 array of the Pole coordinates, or None entries if coordinates
        are undefined.

    Methods
    -------
    line(node: Node = None)
        Returns the slope and intercept [m, n] of the line passing
        through the Pole in the specified direction, or special cases
        for vertical lines. Returns False if the Pole has the same
        location as a node.

    Notes
    -----
    - Pollinie: Line connecting three Poles that defines the relative
      motion of the involved bodies. Movable systems always have at
      least three Poles on a Pollinie.
    - Absolute Pollinie: Connects two absolute Poles and the relative
      Pole of the connected bodies, e.g., (1)-(1/2)-(2).
    - Relative Pollinie: Connects three relative Poles, e.g.,
      (1/2)-(2/3)-(1/3), defining relative motion among the three
      bodies.

    References
    ----------
    .. [1] D. Dinkler. "Grundlagen der Baustatik: Modelle und
           Berechnungsmethoden für ebene Stabtragwerke". Band 1,
           pp. 95 ff., 2011.
    """
    node: Node
    same_location: bool = False
    direction: float = None
    is_infinite: bool = False

    def __post_init__(self):
        if self.is_infinite:
            if self.direction is None:
                raise ValueError("A Pole at infinity must have a direction.")
        elif not self.same_location:
            if self.direction is None:
                raise ValueError(
                    "A Pole with unknown coordinates must have a direction.")
        else:
            if self.direction is not None:
                raise ValueError(
                    "A Pole with same_location=True cannot have a direction.")

    def __eq__(self, other):
        return (isinstance(other, Pole)
                and self.x == other.x and self.z == other.z)

    def __hash__(self):
        return hash((self.x, self.z))

    @property
    def x(self):
        return self.node.x if self.same_location else None

    @property
    def z(self):
        return self.node.z if self.same_location else None

    @property
    def coords(self):
        return np.array([[self.x], [self.z]])

    def line(self, node: Node = None):
        if self.same_location:
            return False

        if node is not None:
            x, z = node.x, node.z
        else:
            x, z = self.node.x, self.node.z

        if np.isclose(np.cos(self.direction), 0, atol=1e-9):
            return None, x
        elif np.isclose(np.cos(self.direction), -1, atol=1e-9):
            return 0, z

        m = np.tan(self.direction)

        n = z - m * x
        return [m, n]

    def __str__(self):
        if self.is_infinite:
            return f"Pole(∞, dir={np.degrees(self.direction):.1f}°)"
        elif self.same_location:
            return f"Pole({self.node.x}|{self.node.z})"
        else:
            return (f"Pole(({self.node.x}|{self.node.z}), dir"
                    f"={np.degrees(self.direction):.1f}°)")


@dataclass(eq=False)
class Chain:
    """
    Represents a kinematic chain of bars and poles in a planar structural
    system (Polplan).

    A Chain corresponds to a "Scheibe" — a kinematically rigid subsystem.
    Chains are composed of bars connected at nodes and are associated with
    absolute and relative poles that define rotational and translational
    motions. Chains can be used to determine Pollinien and relative motion
    vectors in a Poleplan.

    Attributes
    ----------
    bars : set of Bar
        Set of bars that belong to this chain. Must contain at least one
        bar.
    relative_pole : set of Pole
        Set of relative poles associated with this chain, defining relative
        rotations between connected bodies.
    absolute_pole : Pole or None
        Absolute (fixed) pole of the chain, if known. Defines the rotation
        center of the chain.
    connection_nodes : set of Node
        Nodes that connect bars of the chain.

    _stiff : bool, default=False
        Internal flag indicating whether the chain is kinematically stiff.
    _angle_factor : float, default=0
        Internal factor used for calculating rotation angles.
    _angle : float, default=0
        Current rotation angle of the chain.

    Properties
    ----------
    solved : bool
        True if the chain is fully solved (all poles known or chain stiff).
    solved_absolute_pole : bool
        True if the absolute pole is fully defined (coordinates or infinite).
    solved_relative_pole : bool
        True if all relative poles are fully defined (coordinates or infinite).
    angle : float
        Rotation angle of the chain.
    angle_factor : float
        Factor applied to calculate rotations.
    stiff : bool
        Indicates whether the chain is kinematically stiff.
    nodes : list of Node
        List of nodes that belong to the chain.
    apole_lines : dict
        Dictionary of absolute polelines for all relative poles.
    displacement_to_rpoles : dict or None
        Vectors from the absolute pole to each relative pole, or None if the
        absolute pole is at infinity.

    Methods
    -------
    add_connection_node(node)
        Add a single node or a set of nodes to the chain's connection nodes.
    set_absolute_pole(pole, overwrite=False)
        Set or update the absolute pole of the chain.
    add_relative_pole(pole)
        Add one or multiple relative poles to the chain.
    add_bars(bars)
        Add multiple bars to the chain.

    Notes
    -----
    - Scheibe: A kinematically rigid subsystem. In a system of connected
      Scheiben, multiple chains are hinge-connected but cannot translate
      relative to each other.
    - Pollinien: Lines connecting poles that define relative motion among
      chains. Chains with multiple relative poles may have multiple
      absolute pollinien.
    - The chain's stiffness (_stiff) becomes True if absolute and relative
      poles are incompatible or overdetermined.

    References
    ----------
    .. [1] D. Dinkler. "Grundlagen der Baustatik: Modelle und
           Berechnungsmethoden für ebene Stabtragwerke". Band 1,
           pp. 95 ff., 2011.
    """

    bars: set = field(default_factory=set)
    relative_pole: set = field(default_factory=set)
    absolute_pole: Pole = None
    connection_nodes: set = field(default_factory=set)

    _stiff: bool = field(init=False, default=False)
    _angle_factor: float = field(init=False, default=0)
    _angle: float = field(init=False, default=0)

    def __post_init__(self):
        if len(self.bars) == 0:
            raise ValueError('There need to be at least one bar.')

    @property
    def solved(self) -> bool:
        # if self.solved_absolute_pole:
        #     self.solved_relation_aPole_rPole
        if self.stiff:
            return True
        poles_valid = (
                self.solved_absolute_pole
                and self.solved_relative_pole
        )
        return (
                self.bars is not None
                and len(self.connection_nodes) > 0
                and poles_valid
        )

    @property
    def solved_absolute_pole(self) -> bool:
        return (
                self.absolute_pole is not None
                and (
                        self.absolute_pole.same_location
                        or self.absolute_pole.is_infinite
                )
        )

    @property
    def solved_relative_pole(self) -> bool:
        return all(pole.same_location or pole.is_infinite
                   for pole in self.relative_pole)

    # Other properties
    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Angle must be a number")
        self._angle = float(value)

    @property
    def angle_factor(self):
        return self._angle_factor

    @angle_factor.setter
    def angle_factor(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Angle factor must be a number")
        self._angle_factor = float(value)

    @property
    def stiff(self):
        return self._stiff

    @stiff.setter
    def stiff(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Stiff must be bool.")
        self._stiff = bool(value)

    @property
    def nodes(self):
        return list(
            {node for bar in self.bars for node in (bar.node_i, bar.node_j)}
        )

    def add_connection_node(self, node: Node | set[Node]):
        if isinstance(node, Node):
            self.connection_nodes.add(node)
        elif isinstance(node, set):
            self.connection_nodes.update(node)
        else:
            raise TypeError("Expected a Node or a list of Nodes")

    def set_absolute_pole(self, pole: Pole, overwrite: bool = False):
        if not isinstance(pole, Pole):
            raise TypeError("absolute_pole must be an instance of Pole.")

        if self.absolute_pole is not None:
            if overwrite:
                self.absolute_pole = pole
            else:
                from sstatics.core.solution.poleplan.operation import (
                    get_intersection_point, validate_point_on_line
                )
                if (self.absolute_pole.same_location and
                        pole.same_location):
                    # Fall 1: Vorhandener Pol hat Punkt, neuer Pol hat Punkt
                    #  -> Vergleich der Punkte
                    if self.absolute_pole.node != pole.node:
                        self.stiff = True
                    self.absolute_pole = Pole(pole.node,
                                              same_location=True)
                elif (self.absolute_pole.same_location and
                      not pole.same_location):
                    # Fall 2: Vorhandener Pol hat Punkt, neuer Pol hat Linie
                    #  -> Punkt liegt auf Linie
                    point = (self.absolute_pole.x, self.absolute_pole.z)
                    if not validate_point_on_line(pole.line(), point):
                        self.stiff = True
                    self.absolute_pole = Pole(self.absolute_pole.node,
                                              same_location=True)
                elif (not self.absolute_pole.same_location and
                      pole.same_location):
                    # Fall 3: Vorhandener Pol hat Linie, neuer Pol hat Punkt
                    #  -> Punkt liegt auf Linie
                    point = [pole.x, pole.z]
                    if not validate_point_on_line(self.absolute_pole.line(),
                                                  point):
                        self.stiff = True
                    self.absolute_pole = Pole(pole.node,
                                              same_location=True)
                elif (not self.absolute_pole.same_location and
                      not pole.same_location):
                    # Fall 4: Vorhandener Pol hat Linie, neuer Pol hat Linie
                    #  -> Schnittpunkt der Linien
                    line_1 = self.absolute_pole.line()
                    line_2 = pole.line()
                    x, z = get_intersection_point(line_1, line_2)
                    if x is not None:
                        if x != float('inf'):
                            if pole.node.x == x and pole.node.z == z:
                                self.absolute_pole = Pole(
                                    pole.node,
                                    same_location=True)
                            elif (self.absolute_pole.node.x == x and
                                  self.absolute_pole.node.z == z):
                                self.absolute_pole = Pole(
                                    self.absolute_pole.node,
                                    same_location=True)
                            else:
                                self.absolute_pole = Pole(
                                    Node(x=x, z=z),
                                    same_location=True)
                    else:
                        self.absolute_pole = Pole(
                            Node(x=pole.node.x,
                                 z=pole.node.z),
                            same_location=False, is_infinite=True,
                            direction=pole.direction)
        else:
            self.absolute_pole = pole

    def add_relative_pole(self, pole: Pole | list[Pole]):
        if isinstance(pole, Pole):
            self.relative_pole.add(pole)
        elif isinstance(pole, set):
            if not all(isinstance(p, Pole) for p in pole):
                raise TypeError(
                    "All items in the list must be instances of Pole.")
            self.relative_pole.update(pole)
        else:
            raise (
                TypeError("Expected a Pole object or a list of Pole objects."))

    def add_bars(self, bars: set[Bar]):
        self.bars.update(bars)

    @property
    def apole_lines(self):
        # kann nur aufgestellt werden,
        #   wenn die Koordinaten des Absolutpols bekannt sind
        if self.absolute_pole.same_location:
            # wenn eine Scheibe mehrere Relativepole besitzt hat sie Scheibe
            # mehrere Absolutpollinien
            #   -> Geradenparameter werden in einer Liste gespeichert
            line_dict = {}
            for rPol in self.relative_pole:
                # Absolutpollinie kann aufgestellt werden,
                #   1. wenn die Koordinaten des Relativpols bekannt sind
                if rPol.same_location:
                    # Vertikale Gerade x = n
                    if rPol.x == self.absolute_pole.x:
                        m, n = None, rPol.x
                    # Allgemeine Gerade y = m * x + n
                    #   (Sonderfall: horizontale Gerade y = n)
                    else:
                        m = ((rPol.z - self.absolute_pole.z) /
                             (rPol.x - self.absolute_pole.x))
                        n = self.absolute_pole.z - m * self.absolute_pole.x
                    line_dict[rPol.node] = [m, n]
                #   2. wenn die Koordinaten des Relativpols im Unendlichen
                #      liegen
                #      -> hierbei wird die Gerade in den Absolutpol verschoben
                else:
                    line_dict[rPol.node] = rPol.line(
                        node=self.absolute_pole.node)
            return line_dict
        else:
            line_dict = {}
            for rPol in self.relative_pole:
                if rPol.same_location:
                    line_dict[rPol.node] = self.absolute_pole.line(node=rPol)
                else:
                    line_dict[rPol.node] = rPol.line(
                        node=self.absolute_pole.node)
            return line_dict

    @property
    def displacement_to_rpoles(self):
        # Stellt einen Vektor zwischen aPole und rPole auf
        #   -> kann nur aufgestellt werden, wenn der aPole nicht im
        #      Unendlichen liegt
        if self.absolute_pole.is_infinite:
            return None
        else:
            aPole_coords = self.absolute_pole.coords
            # wenn eine Scheibe mehrere Relativepole besitzt hat sie Scheibe
            # mehrere Absolutpollinien und damit mehrere Vektoren
            #   -> Vektoren werden in einer Liste gespeichert
            vec_dict = {}

            for rPole in self.relative_pole:
                if rPole.is_infinite:
                    rPole_coords = np.array([[rPole.node.x], [rPole.node.z]])
                    vec_dict[rPole] = (rPole_coords - aPole_coords)
                else:
                    vec_dict[rPole] = (rPole.coords - aPole_coords)
            return vec_dict

    def __str__(self):
        bar_str = ", ".join(
            f"({b.node_i.x}|{b.node_i.z}) — ({b.node_j.x}|{b.node_j.z})"
            for b in self.bars
        )

        conn_str = ", ".join(f"({n.x}|{n.z})" for n in self.connection_nodes)

        # Relative Pole
        if self.relative_pole:
            rPol_str = ", ".join(str(p) for p in self.relative_pole)
        else:
            rPol_str = "None"

        # Absoluter Pol
        aPol_str = str(self.absolute_pole) if self.absolute_pole else "None"

        return (
            f"    bars        : {bar_str}\n"
            f"    conn_nodes  : {conn_str}\n"
            f"    rPol        : {rPol_str}\n"
            f"    aPol        : {aPol_str}\n"
            f"    stiff       : {self.stiff}\n"
        )


@dataclass(eq=False)
class Poleplan(LoggerMixin):
    """
    Represents the Poleplan (Polplan) of a planar structural system.

    The Poleplan captures the kinematic relationships of the system using
    chains (Scheiben), absolute poles, and relative poles. It systematically
    constructs all necessary poles and polelines to describe relative and
    absolute motions of the rigid subsystems.

    Construction Procedure
    ----------------------
    The creation of a Poleplan is performed in several steps [1]_:

    1. Identification and naming of all chains (Scheiben) in the system.
    2. Identification and naming of immediately recognizable absolute poles
       (e.g., at fixed supports) and relative poles (e.g., at moment hinges).
    3. Construction of directly visible polelines, e.g., perpendicular to
       movable supports, normal force, and shear hinges.
    4. Stepwise determination of additional poles using absolute and relative
       polelines. Two geometric conditions (directions or lines) define the
       intersection point of the unknown pole. If polelines are parallel, the
       pole lies at infinity.
    5. Verification of the complete Poleplan for consistency. Contradictions
       arise, e.g., if a chain has multiple absolute poles or if two chains
       share multiple relative poles. Contradictions indicate that the system
       or parts of it are kinematically rigid. If no contradictions exist, the
       system can be considered movable.

    Attributes
    ----------
    system : System
        The structural system being analyzed.
    debug : bool, default=False
        Enable debug logging for intermediate steps.
    chains : list of Chain
        All chains (Scheiben) identified in the system after processing.
    solved : bool
        True if the Poleplan is consistent and all chains/poles are solved.

    _node_to_chains : dict, optional
        Mapping of Node → list of chains that contain the node.
    _node_to_multiple_chains : dict, optional
        Cached mapping of nodes that belong to more than one chain.

    Properties
    ----------
    node_to_chains : dict
        Mapping of nodes to chains containing them. Setter validates input
        and clears cached multiple-chain data.
    node_to_multiple_chains : dict
        Nodes that belong to more than one chain. Computed lazily from
        node_to_chains.

    Methods
    -------
    set_angle(target_chain, target_angle)
        Adjust the rotation angle of a specified chain.
    rigid_motion(n_disc=2)
        Return a list of rigid body displacement objects for plotting.
    get_chain_for(target)
        Return the chain containing the given Bar or Node.
    find_adjacent_chain(node, chain)
        Find a non-stiff adjacent chain sharing a node with the specified
        chain, returning absolute pole coordinates, node coordinates, and
        the adjacent chain.

    Notes
    -----
    - The Poleplan is constructed using a multi-step pipeline involving
      ChainIdentifier, PoleIdentifier, and Validator classes.
    - It captures kinematic constraints and relative motions of rigid
      subsystems.
    - Chains (Scheiben) are rigid but may be hinge-connected and
      non-translating with respect to each other.
    - Absolute poles define fixed rotation centers, while relative poles
      define rotations between chains.
    - Pollinien are lines connecting poles that define relative motion
      geometry.

    References
    ----------
    .. [1] D. Dinkler. "Grundlagen der Baustatik: Modelle und
           Berechnungsmethoden für ebene Stabtragwerke". Band 1,
           pp. 95 ff., 2011.
    """

    system: System
    debug: bool = False

    _node_to_chains: Optional[dict] = field(init=False, default=None)
    _node_to_multiple_chains: Optional[dict] = field(init=False, default=None)
    _set_angle: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Initialise the pole‑plan and run the processing pipeline."""
        self.logger.info("Poleplan initialisation started")
        from sstatics.core.solution.poleplan.operation import (
            ChainIdentifier,
            PoleIdentifier,
            Validator,
        )

        steps: list[Type] = [ChainIdentifier, PoleIdentifier, Validator]
        results = {}

        for i, step_class in enumerate(steps):
            step_name = step_class.__name__
            self.logger.debug(
                f"Running step {i + 1}/{len(steps)}: {step_name}")

            if i == 0:
                # First step receives the system directly.
                step = step_class(self.system, debug=self.debug)
                self.logger.debug("Created first step instance with system")
            else:
                # Subsequent steps receive data from previous results.
                step = step_class(
                    results.get("chains"),
                    results.get("node_to_chains"),
                    debug=self.debug
                )
                self.logger.debug(
                    f"Created {step_name} with previous results: "
                    f"chains={bool(results.get('chains'))}, "
                    f"node_to_chains={bool(results.get('node_to_chains'))}"
                )

            try:
                step()
                self.logger.info(f"Step {step_name} executed successfully")
            except Exception as exc:
                self.logger.error(
                    f"Error while executing step {step_name}: {exc}",
                    exc_info=True,
                )
                raise

            # Store and propagate intermediate results.
            results["chains"] = step.chains
            self.node_to_chains = step.node_to_chains
            results["node_to_chains"] = self.node_to_multiple_chains
            self.logger.debug(
                f"Updated results after {step_name}: "
                f"{len(step.chains)} chains, "
                f"{len(self.node_to_multiple_chains)} multiple‑chain nodes"
            )

            if i == 2:  # Validator step
                results["solved"] = step.solved
                self.logger.debug(f"Validator solved flag: {step.solved}")

        self.chains = results["chains"]
        self.solved = results["solved"]
        self.logger.info(
            "Poleplan initialisation completed: "
            f"{len(self.chains)} total chains, solved={self.solved}"
        )

    @property
    def node_to_chains(self):
        """Mapping of node → list of chains that contain the node."""
        return self._node_to_chains

    @node_to_chains.setter
    def node_to_chains(self, value: dict) -> None:
        """Validate and store the node‑to‑chains dictionary."""
        if not isinstance(value, dict):
            msg = "node_to_chains must be a dict"
            self.logger.error(msg)
            raise TypeError(msg)
        self._node_to_chains = value
        self.logger.debug(
            f"node_to_chains set with {len(value)} entries"
        )
        # Invalidate cached multiple‑chain data.
        self._node_to_multiple_chains = None
        self.logger.debug("Cached node_to_multiple_chains cleared")

    @property
    def node_to_multiple_chains(self) -> dict:
        """Cache of nodes that belong to more than one chain."""
        if self._node_to_multiple_chains is None:
            self._node_to_multiple_chains = self._filter_multiple_chains(
                self.node_to_chains
            )
            self.logger.debug(
                f"Computed node_to_multiple_chains: "
                f"{len(self._node_to_multiple_chains)} nodes"
            )
        return self._node_to_multiple_chains

    @staticmethod
    def _filter_multiple_chains(conn: dict) -> dict:
        """Return only those nodes that appear in > 1 chain."""
        filtered = {k: v for k, v in conn.items() if len(v) > 1}
        return filtered

    def set_angle(self, chain_idx: int = 0, angle: float = 1) -> None:
        """
        Adjust the angle of ``target_chain`` to ``target_angle``.
        """
        self.logger.info(
            f"Starting to set target chain at index {chain_idx} "
            f"to angle {angle}")
        if chain_idx >= len(self.chains) or chain_idx < 0:
            self.logger.error(
                f"Chain index {chain_idx} is out of range. "
                f"Valid indices: 0 to {len(self.chains) - 1}")
            raise IndexError(f"Chain index {chain_idx} is not in self.chains")
        target_chain = self.chains[chain_idx]
        self.logger.info(
            f"Setting angle for chain index: {chain_idx} to {angle}"
        )
        from sstatics.core.solution.poleplan.operation import \
            AngleCalculator

        try:
            angle_calculator = AngleCalculator(
                self.chains, self.node_to_multiple_chains, self.debug
            )
            angle_calculator.calculate_angles(target_chain, angle)
            self.logger.debug("Angle calculation completed")
            self._set_angle = True
        except Exception as exc:
            self.logger.error(
                f"Failed to calculate angles: {exc}",
                exc_info=True,
            )
            raise

    def rigid_motion(self, n_disc: int = 2):
        """
        Return a list of Rigid Body Displacement Objects. They can use for
        plotting.
        """
        self.logger.info("Generating displacement vector for each bar")
        from sstatics.core.solution.poleplan.operation import (
            DisplacementCalculator,
        )
        from sstatics.core.postprocessing.results import RigidBodyDisplacement
        try:
            fig = DisplacementCalculator(
                self.chains, self.system.bars, self.node_to_multiple_chains,
                debug=self.debug
            )()
            self.logger.info("Generating for each bar a "
                             "rigid-body-displacement-object for plotting.")
            # creating rigid-body-displacement-object for plotting
            rbd_objects = []
            for i, bar in enumerate(self.system.bars):
                rdb = RigidBodyDisplacement(
                    bar=bar,
                    deform=fig[i],
                    n_disc=n_disc
                )
                rbd_objects.append(rdb)

            self.logger.debug("Displacement figure created")
            return rbd_objects
        except Exception as exc:
            self.logger.error(
                f"Displacement calculation failed: {exc}",
                exc_info=True,
            )
            raise

    def fig(self, n_disc: int = 2):
        """
        Return a list of Rigid Body Displacement Objects. They can use for
        plotting.
        """
        self.logger.info("Generating displacement vector for each bar")
        from sstatics.core.solution.poleplan.operation import (
            DisplacementCalculator,
        )
        from sstatics.core.postprocessing.results import RigidBodyDisplacement
        try:
            fig = DisplacementCalculator(
                self.chains, self.system.bars, self.node_to_multiple_chains,
                debug=self.debug
            )()
            self.logger.info("Generating for each bar a "
                             "rigid-body-displacement-object for plotting.")
            # creating rigid-body-displacement-object for plotting
            rbd_objects = []
            for i, bar in enumerate(self.system.bars):
                rdb = RigidBodyDisplacement(
                    bar=bar,
                    deform=fig[i],
                    n_disc=n_disc
                )
                rbd_objects.append(rdb)

            self.logger.debug("Displacement figure created")
            return fig
        except Exception as exc:
            self.logger.error(
                f"Displacement calculation failed: {exc}",
                exc_info=True,
            )
            raise

    def get_chain_for(self, target: Bar | Node) -> Optional[Chain]:
        """
        Returns the chain in which the given bar or node is contained.
        """
        try:
            if isinstance(target, Bar):
                self.logger.debug(
                    f"Searching for chain containing bar {target}")
                chain = next(
                    (c for c in self.chains if target in c.bars), None
                )
            elif isinstance(target, Node):
                self.logger.debug(f"Looking up chain for node {target}")
                chain = next(
                    (
                        c for c in self.chains
                        if any(target.same_location(n) for n in
                               c.connection_nodes)
                    ),
                    None,
                )
            else:
                raise TypeError(
                    f"Target must be a Bar or Node, got {type(target)}")

            if chain:
                self.logger.debug(f"Found chain {chain} for target {target}")
            else:
                self.logger.warning(f"No chain found for target {target}")

            return chain

        except Exception as exc:
            self.logger.error(
                f"Error while searching for chain for {target}: {exc}",
                exc_info=True)
            raise

    def find_adjacent_chain(self, node, chain):
        # TODO: Refactor this method and get_chain_and_angle(…) of
        #  InfluenceLine class together.
        """
        Find a non‑stiff adjacent chain that shares ``node`` with ``chain``.
        Returns ``(absolute_pole_coords, node_coords, adjacent_chain)`` or
        ``(None, None, None)`` if none is found.
        """
        self.logger.debug(
            f"Searching adjacent chains for node {node} and chain {chain}"
        )
        conn_chain = self.node_to_multiple_chains.get(node, [])

        if len(conn_chain) <= 1:
            self.logger.info(
                "Node is connected to zero or one chain – no adjacency "
                "possible"
            )
            return None, None, None

        self.logger.info(
            f"Node {node} is shared by {len(conn_chain)} chains"
        )
        for c in conn_chain:
            if c == chain:
                continue  # skip the original chain
            self.logger.debug(f"Evaluating adjacent chain {c}")

            if c.stiff:
                self.logger.debug("Chain is stiff – skipping")
                continue

            # Gather absolute pole coordinates.
            aPole_coords = c.absolute_pole.coords
            self.logger.debug(
                f"Absolute pole coordinates: {aPole_coords.tolist()}"
            )

            # Determine relative pole coordinates.
            for rPole in c.relative_pole:
                if not rPole.is_infinite:
                    node_coords = rPole.coords
                    self.logger.debug(
                        f"Finite relative pole found: {node_coords.tolist()}"
                    )
                else:
                    node_coords = aPole_coords = np.array(
                        [[rPole.node.x], [rPole.node.z]]
                    )
                    self.logger.debug(
                        "Infinite relative pole – using absolute coordinates"
                    )
                return aPole_coords, node_coords, c

        self.logger.warning("No suitable adjacent non‑stiff chain found")
        return None, None, None

    def plot(self, mode: str = 'MPL'):
        from sstatics.graphic_objects.poleplan import PoleplanGraphic

        if not self._set_angle:
            self.set_angle()

        PoleplanGraphic(poleplan=self).show()
