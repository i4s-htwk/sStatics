
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional
import numpy as np

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


@dataclass(eq=False)
class Chain:

    bars: set = field(default_factory=set)
    relative_pole: set = field(default_factory=set)
    absolute_pole: Pole = None
    connection_nodes: set = field(default_factory=set)

    _stiff: bool = False
    _angle_factor: float = 0
    _angle: float = 0

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


@dataclass(eq=False)
class Poleplan:

    system: System

    def __post_init__(self):
        self.bars = self.system.bars
        self.chains = []

        from sstatics.core.preprocessing.poleplan.operation import (
            ChainIdentifier, PoleIdentifier, Validator
        )

        self.chain_identifier = ChainIdentifier(
            self.system, self.bars, self.chains)()

        self.pole_identifier = PoleIdentifier(
            self.chains, self.node_to_chain_map, self.bars)()

        self.solved = Validator(self.chains, self.node_to_chain_map)()

    @cached_property
    def node_to_chain_map(self):
        conn = {}
        for chain in self.chains:
            for n in chain.connection_nodes:
                if n not in conn:
                    conn[n] = []
                conn[n].append(chain)
            for bar in chain.bars:
                for node in (bar.node_i, bar.node_j):
                    if node not in chain.connection_nodes and node in conn:
                        chain.add_connection_node(node)
        return {key: chains for key, chains in conn.items() if len(chains) > 1}

    def set_angle(self, target_chain, target_angle):
        from sstatics.core.preprocessing.poleplan.operation import (
            AngleCalculator
        )
        angle_calculator = AngleCalculator(self.chains, self.node_to_chain_map)
        angle_calculator.set_angle(target_chain, target_angle)

    def get_displacement_figure(self):
        from sstatics.core.preprocessing.poleplan.operation import (
            DisplacementCalculator
        )
        return DisplacementCalculator(
            self.chains, self.bars, self.node_to_chain_map
        )()

    def get_chain(self, bars: set[Bar]) -> Optional[Chain]:
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
        """Returns the chain that contains any of the given bars."""
        return next(
            (chain for chain in self.chains if bars & chain.bars), None
        )

    def get_chain_node(self, node) -> Optional[Chain]:
        """Returns the chain that is connected to the given node."""
        return next(
            (chain for chain in self.chains
             if any(node.same_location(n)
                    for n in chain.connection_nodes)), None
        )

    def find_adjacent_chain(self, node, chain):
        # TODO: Refactor this method and get_chain_and_angle(…) of
        #  InfluenceLine class together.
        conn_chain = self.node_to_chain_map.get(node, [])

        if len(conn_chain) > 1:
            print(' -> es gibt angrenzende Scheiben')

            print('Gibt es Starre Scheiben?')
            for c in conn_chain:
                if c == chain:
                    continue
                print(
                    self.chains.index(c))
                if c.stiff:
                    print('starr')
                    continue

                # Falls nicht starr,
                # die absoluten Koordinaten speichern
                aPole_coords = c.absolute_pole.coords

                # Die relative Pol-Koordinate ermitteln,
                # falls der Knoten nicht übereinstimmt
                for rPole in c.relative_pole:
                    if not rPole.is_infinite:
                        node_coords = rPole.coords
                    else:
                        node_coords = aPole_coords = np.array([
                            [rPole.node.x],
                            [rPole.node.z]
                        ])
                    return aPole_coords, node_coords, c
        return None, None, None
