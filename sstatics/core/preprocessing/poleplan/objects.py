
from dataclasses import dataclass, field
from typing import Optional, Type
import numpy as np

from sstatics.core.logger_mixin import LoggerMixin
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System


@dataclass(eq=False)
class Pole:

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

        # Prüfen, ob die Gerade vertikal ist (cos(direction) ≈ 0)
        if np.isclose(np.cos(self.direction), 0, atol=1e-9):
            return None, x  # Vertikale Gerade: x = x0
        elif np.isclose(np.cos(self.direction), -1, atol=1e-9):
            return 0, z

        # Steigung m = tan(direction)
        m = np.tan(self.direction)

        # y-Achsenabschnitt n berechnen: z = m*x + n → n = z - m*x
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

    # @property
    # def solved_relation_aPole_rPole(self):
    #     if self.absolute_pole.is_infinite:
    #         for rPole in self.relative_pole:
    #             print(rPole)
    #             if rPole.same_location:
    #                 if not validate_point_on_line(
    #                         self.absolute_pole.line(), (rPole.x, rPole.z)):
    #                     self.stiff = True
    #                     return False
    #     return True

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
                from sstatics.core.preprocessing.poleplan.operation import (
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

    system: System
    debug: bool = False

    _node_to_chains: Optional[dict] = field(init=False, default=None)
    _node_to_multiple_chains: Optional[dict] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialise the pole‑plan and run the processing pipeline."""
        self.logger.info("Poleplan initialisation started")
        from sstatics.core.preprocessing.poleplan.operation import (
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

    def set_angle(self, target_chain, target_angle) -> None:
        """
        Adjust the angle of ``target_chain`` to ``target_angle``.
        """
        self.logger.info(
            f"Setting angle for chain \n {target_chain} to {target_angle}"
        )
        from sstatics.core.preprocessing.poleplan.operation import \
            AngleCalculator

        try:
            angle_calculator = AngleCalculator(
                self.chains, self.node_to_multiple_chains
            )
            angle_calculator.calculate_angles(target_chain, target_angle)
            self.logger.debug("Angle calculation completed")
        except Exception as exc:
            self.logger.error(
                f"Failed to calculate angles: {exc}",
                exc_info=True,
            )
            raise

    def get_displacement_figure(self):
        """
        Return a matplotlib figure (or similar) visualising displacement
        of the pole‑plan.
        """
        self.logger.info("Generating displacement figure")
        from sstatics.core.preprocessing.poleplan.operation import (
            DisplacementCalculator,
        )
        try:
            fig = DisplacementCalculator(
                self.chains, self.system.bars, self.node_to_multiple_chains
            )()
            self.logger.debug("Displacement figure created")
            return fig
        except Exception as exc:
            self.logger.error(
                f"Displacement calculation failed: {exc}",
                exc_info=True,
            )
            raise

    # def get_chain(self, bars: set[Bar]) -> Optional[Chain]:
    #     # TODO: Simplify get_chain to accept a single Bar object.
    #     #   Original:
    #     #     def get_chain(self, bars: set[Bar]) -> Optional[Chain]:
    #     #         """Returns chain containing any given bars."""
    #     #         return next((chain for chain in self.chains
    #     #                      if bars & chain.bars), None)
    #     #   Simplified:
    #     #     def get_chain(self, bar: Bar) -> Optional[Chain]:
    #     #         """Returns chain containing the given bar."""
    #     #         return next((chain for chain in self.chains
    #     #                      if bar in chain.bars), None)
    #     #   Add a separate method for multiple Bar objects:
    #     #     def get_chain_containing_any(self, bars: set[Bar]):
    #     #         """Returns chain containing any given bars."""
    #     #         return next((chain for chain in self.chains
    #     #                      if bars & chain.bars), None)
    #     """Returns the chain that contains any of the given bars."""
    #     return next(
    #         (chain for chain in self.chains if bars & chain.bars), None
    #     )

    def get_chain(self, bars: set[Bar]) -> Optional[Chain]:
        """
        Returns the chain that contains any of the given bars.
        """
        # TODO: Simplify get_chain to accept a single Bar object.
        #   Original:
        #     def get_chain(self, bars: set[Bar]) -> Optional[Chain]:
        #         """Returns chain containing any given bars."""
        #         return next((chain for chain in self.chains
        #                      if bars & chain.bars), None)
        #   Simplified:
        #     def get_chain(self, bar: Bar) -> Optional[Chain]:
        #         """Returns chain containing the given bar."""
        #         return next((chain for chain in self.chains
        #                      if bar in chain.bars), None)
        #   Add a separate method for multiple Bar objects:
        #     def get_chain_containing_any(self, bars: set[Bar]):
        #         """Returns chain containing any given bars."""
        #         return next((chain for chain in self.chains
        #                      if bars & chain.bars), None)
        self.logger.debug(
            f"Searching for chain containing any of {len(bars)} bars"
        )
        try:
            chain = next(
                (c for c in self.chains if bars & c.bars), None
            )
            if chain:
                self.logger.debug(
                    f"Found chain {chain} containing the bars"
                )
            else:
                self.logger.warning("No chain contains the supplied bars")
            return chain
        except Exception as exc:
            self.logger.error(
                f"Error while searching for chain: {exc}",
                exc_info=True,
            )
            raise

    def get_chain_node(self, node) -> Optional[Chain]:
        """
        Returns the chain that is connected to the given node.
        """
        self.logger.debug(f"Looking up chain for node {node}")
        try:
            chain = next(
                (
                    c
                    for c in self.chains
                    if any(node.same_location(n) for n in c.connection_nodes)
                ),
                None,
            )
            if chain:
                self.logger.debug(f"Node {node} belongs to chain {chain}")
            else:
                self.logger.warning(f"No chain found for node {node}")
            return chain
        except Exception as exc:
            self.logger.error(
                f"Failed to locate chain for node: {exc}",
                exc_info=True,
            )
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
