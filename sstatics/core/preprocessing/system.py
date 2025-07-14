
from collections import defaultdict
from dataclasses import dataclass, replace, field
from functools import cache, cached_property
from typing import Literal, Dict, Callable, List, Optional
from itertools import combinations

import numpy as np
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.loads import (
    BarPointLoad, NodePointLoad
)
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.temperature import BarTemp
from sstatics.core.utils import (
    get_intersection_point, validate_point_on_line
)


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
    stiff: bool = False
    angle_factor: float = 0
    angle: float = 0

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
    def absolute_pole_lines_dict(self):
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

    # Drehwinkel
    @property
    def vec_aPole_rPole_dict(self):
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

    def set_angle_factor(self, factor):
        self.angle_factor = factor

    def set_angle(self, angle):
        self.angle = angle


@dataclass(eq=False)
class System:
    """Represents a statical system composed of interconnected bars.

    This class models a mechanical system made up of interconnected bars, where
    each bar connects two nodes.

    Parameters
    ----------
    bars : tuple[Bar, ...] | list[Bar]
        A list or tuple of bars that define the structure of the statical
        system. The bars must be provided in a consecutive order, as they
        represent a connected structure.

    Raises
    ------
    ValueError
        Raised if any of the following conditions are met:

        * The list of bars is empty.
        * Two or more bars share the same geometric location.
        * Two distinct node instances occupy the same spatial position.
        * The system's connectivity graph is not fully connected.

    Attributes
    ----------
    segmented_bars : tuple[Bar, ...]
        The bars of the system after segmentation.

    dof : int
        Degrees of freedom for the system (fixed at 3).
    """

    bars: tuple[Bar, ...] | list[Bar]
    user_divisions: Optional[Dict[Bar, List[float]]] = None

    # weitere Validierungen? sich schneidende Stäbe?
    def __post_init__(self):
        self.bars = tuple(self.bars)
        if len(self.bars) == 0:
            raise ValueError('There need to be at least one bar.')
        for i, bar in enumerate(self.bars[0:-1]):
            if any([
                bar.same_location(other_bar) for other_bar in self.bars[i + 1:]
            ]):
                raise ValueError(
                    'Cannot instantiate a system with bars that share the '
                    'same location.'
                )
        nodes = self.nodes(segmented=False)
        for i, node in enumerate(nodes[0:-1]):
            for other_node in nodes[i + 1:]:
                if node.same_location(other_node) and node != other_node:
                    raise ValueError(
                        'Inconsistent system. Nodes with the same location '
                        'need to be the same instance.'
                    )
        to_visit, visited = [nodes[0]], []
        while to_visit:
            current_node = to_visit.pop(0)
            if current_node not in visited:
                visited.append(current_node)
                to_visit += self.connected_nodes(segmented=False)[current_node]
        if set(visited) != set(nodes):
            raise ValueError("The system's graph needs to be connected.")

        self.create_mesh(user_divisions=self.user_divisions)

    @property
    def mesh(self):
        return self._mesh.mesh

    @cache
    def connected_nodes(self, segmented: bool = True):
        bars = self.mesh if segmented else self.bars
        connections = {}
        for bar in bars:
            for node in (bar.node_i, bar.node_j):
                if node not in connections:
                    connections[node] = set()
            connections[bar.node_i].add(bar.node_j)
            connections[bar.node_j].add(bar.node_i)
        return {
            node: list(connected_nodes)
            for node, connected_nodes in connections.items()
        }

    @cache
    def node_to_bar_map(self, segmented: bool = True):
        bars = self.mesh if segmented else self.bars
        node_connection = {}
        for bar in bars:
            for node in (bar.node_i, bar.node_j):
                if node not in node_connection:
                    node_connection[node] = []
                node_connection[node].append(bar)
        return node_connection

    @cache
    def nodes(self, segmented: bool = True):
        return list(self.connected_nodes(segmented=segmented).keys())

    def get_polplan(self):
        self.polplan = Polplan(self)

    def create_mesh(self, user_divisions=None):
        self._mesh = MeshGenerator(bars=self.bars,
                                   user_divisions=user_divisions)()


@dataclass(eq=False)
class MeshGenerator:
    bars: List[Bar]
    user_divisions: Optional[Dict[Bar, List[float]]] = None

    def __post_init__(self):
        self.bars = list(self.bars)
        self._mesh: List[Bar] = []
        self._user_segments: List[Bar] = []

    @property
    def mesh(self):
        return self._mesh

    @property
    def user_segments(self):
        return self._user_segments

    def generate(self):
        user_divisions = self.user_divisions or {}
        calc_mesh = []
        user_mesh = []

        for i, bar in enumerate(self.bars):

            user_pos = user_divisions.get(bar, [])
            load_pos = self._get_point_loads(bar)

            if not user_pos and not load_pos:
                calc_mesh.append(bar)
                user_mesh.append(bar)
                continue

            if user_pos:
                user_pos, load_pos = (
                    self._transfer_loads_to_user_pos(user_pos, load_pos)
                )
                user_segments = self._split(bar, user_pos)
            else:
                user_segments = [bar]

            user_mesh.extend(user_segments)

            calc_segments = user_segments.copy()

            if load_pos:
                for idx, pos_load in (
                        self._assign_loads_to_segments(
                            load_pos, user_pos).items()
                ):
                    new_segments = self._split(calc_segments[idx], pos_load)
                    calc_segments[idx:idx + 1] = new_segments

            calc_mesh.extend(calc_segments)

        self._user_segments = user_mesh
        self._mesh = calc_mesh

    @staticmethod
    def _assign_loads_to_segments(load_pos_i, user_pos_i):
        user_pos_i = sorted(set(user_pos_i) | {0.0, 1.0})
        index = {}

        for p, loads in load_pos_i.items():
            for i, (a, b) in enumerate(zip(user_pos_i, user_pos_i[1:])):
                if p in (0.0, 1.0):
                    index.setdefault(
                        i, defaultdict(list))[p].extend(loads)
                    break
                if a < p < b:
                    new_pos = (p - a) / (b - a)
                    index.setdefault(
                        i, defaultdict(list))[new_pos].extend(loads)
                    break
        return index

    @staticmethod
    def _get_point_loads(bar):
        return {load.position: [load] for load in bar.point_loads}

    @staticmethod
    def _transfer_loads_to_user_pos(user_pos_i: list[float], load_pos_i):
        user_positions = defaultdict(list, {k: [] for k in user_pos_i})
        for pos in list(load_pos_i):
            if pos in user_positions or pos in (0.0, 1.0):
                user_positions[pos].extend(load_pos_i[pos])
                del load_pos_i[pos]

        return user_positions, load_pos_i

    @staticmethod
    def _add_end_loads(user_positions, bar):
        for load in bar.point_loads:
            if load.position in (0.0, 1.0):
                user_positions[load.position].append(load)
        return user_positions

    @staticmethod
    def _to_node_load(point_loads):
        return [
            NodePointLoad(load.x, load.z, load.phi, load.rotation)
            for load in point_loads
        ]

    @staticmethod
    def _interp_coords(bar, position: float):
        c, s = np.cos(bar.inclination), np.sin(bar.inclination)
        return (
            bar.node_i.x + c * position * bar.length,
            bar.node_i.z - s * position * bar.length
        )

    @staticmethod
    def _interp_lloads(bar: Bar,
                       prev_bar: Optional[Bar],
                       pos: Optional[float] = None):
        if not bar.line_loads:
            return []
        return [
            replace(
                load,
                pi=(prev_bar.line_loads[i].pj if prev_bar else load.pi),
                pj=(load.pi + (load.pj - load.pi) * pos
                    if pos is not None else load.pj)
            )
            for i, load in enumerate(bar.line_loads)
        ]

    def _split(self,
               bar: Bar,
               pos_dict: Dict[float, List[NodePointLoad]]) -> List[Bar]:
        bars = []
        prev_bar = None

        for pos in sorted(pos_dict):
            if pos in (0.0, 1.0):
                continue

            node_j = self._node(bar, pos, pos_dict[pos])

            if prev_bar is None:
                new_bar = self._bar_first(bar, node_j, pos, pos_dict[0.0])
            else:
                new_bar = self._bar_middle(bar, prev_bar, node_j, pos)

            bars.append(new_bar)
            prev_bar = new_bar

        bars.append(self._bar_last(bar, prev_bar, pos_dict[1.0]))

        return bars

    def _node(self, bar, position: float, point_loads: List[NodePointLoad]):
        x, z = self._interp_coords(bar, position)
        return Node(x, z, loads=self._to_node_load(point_loads))

    def _bar_first(self, bar, node_j, pos, end_point_loads):
        return replace(
            bar,
            node_i=bar.node_i,
            node_j=node_j,
            point_loads=end_point_loads,
            line_loads=self._interp_lloads(bar=bar, prev_bar=None, pos=pos),
            hinge_u_j=False,
            hinge_w_j=False,
            hinge_phi_j=False
        )

    def _bar_middle(self, bar, prev_bar, node_j, pos):
        return replace(
            bar,
            node_i=prev_bar.node_j,
            node_j=node_j,
            point_loads=[],
            line_loads=self._interp_lloads(
                bar=bar, prev_bar=prev_bar, pos=pos
            ),
            hinge_u_i=False,
            hinge_w_i=False,
            hinge_phi_i=False,
            hinge_u_j=False,
            hinge_w_j=False,
            hinge_phi_j=False
        )

    def _bar_last(self, bar, prev_bar, end_point_loads):
        return replace(
            bar,
            node_i=prev_bar.node_j,
            node_j=bar.node_j,
            point_loads=end_point_loads,
            line_loads=self._interp_lloads(bar=bar, prev_bar=prev_bar),
            hinge_u_i=False,
            hinge_w_i=False,
            hinge_phi_i=False
        )

    def __call__(self):
        self.generate()
        return self


@dataclass(eq=False)
class Polplan:

    system: System

    def __post_init__(self):
        self.bars = self.system.bars
        self.chains = []
        nodes = self.system.nodes(segmented=False)
        to_visit, visited = [nodes[0]], []
        print('-----------------------------------------------')
        print('Schritt 1: Scheibenidentifikation + Pole finden')
        print('-----------------------------------------------')
        while to_visit:
            current_node = to_visit.pop(0)
            if current_node not in visited:
                visited.append(current_node)
                to_visit += (
                    self.system.connected_nodes(segmented=False))[current_node]
                self._identify_chains(current_node)
        print('----------------------------')
        print('Schnitt 1.1: Dreiecke finden')
        print('----------------------------')
        while self._identify_triangle():
            continue
        print('------------------------')
        print('Schritt 1 abgeschlossen!')
        print('------------------------')
        print('---------------------------------------------')
        print('Schritt 2: weitere Absolutpole identifizieren')
        print('---------------------------------------------')
        self._identify_pole()

        print('------------------------')
        print('Schritt 2 abgeschlossen!')
        print('------------------------')

        self.print_chains()

        print('------------------------------------------')
        print('Schritt 3: Polplan auf Widersprüche prüfen')
        print('------------------------------------------')

        if self.solved:
            print('Polplan widerspruchsfrei')
            print('------------------------')
            print('Schritt 3 abgeschlossen!')
            print('------------------------')
        else:
            print('Polplan hat Widerspruch')
            print('------------------------')
            print('Schritt 3 abgeschlossen!')
            print('------------------------')

    # Scheiben finden
    def _identify_chains(self, current_node: Node):
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f'({current_node.x},{current_node.z})')
        unassigned_bars = []
        connected_bars = (
            self.system.node_to_bar_map(segmented=False))[current_node]

        for bar in connected_bars:
            print('---------------------------------------------')
            print(f'Stabnr: {self.bars.index(bar)}')
            hinge_check = any([
                bar.hinge_u_i, bar.hinge_w_i, bar.hinge_phi_i
            ]) if bar.node_i == current_node else any([
                bar.hinge_u_j, bar.hinge_w_j, bar.hinge_phi_j
            ])
            if hinge_check:
                print(' -> Gelenk')
                print('   -> hier beginnt eine neue Scheibe')
                self._new_chain({bar}, current_node)
            else:
                print(' -> kein Gelenk')
                unassigned_bars.append(bar)
                print(f'   -> Stab wird zur Stabliste hinzugefügt: '
                      f'{[self.bars.index(b) for b in unassigned_bars]}')
        print('Gibt es noch unverarbeitete angrenzende Stäbe?')
        if unassigned_bars:
            print(f' -> Ja, folgende Stäbe sind noch nicht zugewiesen: '
                  f'{[self.bars.index(b) for b in unassigned_bars]}')
            self._new_chain(set(unassigned_bars), current_node)

    def _new_chain(self, bars: set[Bar], current_node: Node):
        print(f'Sind Stäben: {[self.bars.index(b) for b in bars]}, '
              f'schon in einer Scheibe vorhanden?')
        existing_chain = self._get_chain(bars)
        if existing_chain:
            print(f'-> Stab {[self.bars.index(b) for b in bars]} '
                  f'schon in Chain-nr.: '
                  f'{self.chains.index(existing_chain)}')
            print(f' -> Stäbe werden Scheibe '
                  f'{self.chains.index(existing_chain)} hinzugefügt')
            self._add_bar_to_chain(existing_chain, bars)
            self._add_node_to_chain(existing_chain, current_node)
        else:
            print(f'-> Stab {[self.bars.index(b) for b in bars]} '
                  f'noch nicht in Chains')
            print(' -> neue Scheibe')
            new_chain = Chain(bars)
            self._add_node_to_chain(new_chain, current_node)
            self.chains.append(new_chain)

        shared_chains = self._find_shared_chains()
        if shared_chains:
            self._merge_chains(shared_chains)
        self._identify_triangle()

    def _add_bar_to_chain(self, chain: Chain, bars: set[Bar]):
        chain.add_bars(bars)

    def _add_node_to_chain(self, chain: Chain, node: Node):
        print('...')
        print(f'Aktueller Knoten ({node.x},{node.z}) '
              f'wird zur Scheibe hinzugefügt')
        # potenzieller Verbindungspunkte zwischen weiteren Scheiben
        connected_bars = self.system.node_to_bar_map(segmented=False)[node]
        chain.add_connection_node(node)

        # Absolutpol oder Linie des Absolutpols
        if node.u == 'fixed' and node.w == 'fixed' and node.phi == 'free':
            print(' -> zweiwertiges Auflager')
            chain.set_absolute_pole(Pole(node, same_location=True))
        elif (node.u == 'fixed' or node.w == 'fixed') and node.phi == 'free':
            print(' -> einwertiges Auflager')
            print('  -> haben angrenzende Stäbe Gelenke')
            for bar in connected_bars:
                # das geht auch nicht? WARUM?

                for end, hinge_phi, hinge_w, hinge_u in [
                    (bar.node_i,
                     bar.hinge_phi_i, bar.hinge_w_i, bar.hinge_u_i),
                    (bar.node_j,
                     bar.hinge_phi_j, bar.hinge_w_j, bar.hinge_u_j),
                ]:
                    if node == end:
                        if bar not in chain.bars:
                            if hinge_w:
                                chain.set_absolute_pole(
                                    Pole(node, same_location=True))
                            continue
                        # if hinge_w or hinge_u and hinge_phi:
                        #     print(f'   -> am Knoten ({node.x}, {node.z}) '
                        #           f'hat Stab {self.bars.index(bar)} '
                        #           f'ein Normal-oder Querkraftgelenk')
                        #     print('     -> wird Relativpol')
                        #     direction = np.pi - bar.inclination if hinge_w \
                        #         else np.pi / 2 - bar.inclination
                        #     chain.add_relative_pole(
                        #         Pole(node,is_infinite=True,
                        #              direction=direction))
                        if hinge_phi:
                            print(f'   -> am Knoten ({node.x}, {node.z}) '
                                  f'hat Stab {self.bars.index(bar)} '
                                  f'ein Momentengelenk')
                            print('     -> wird Relativpol')
                            chain.add_relative_pole(
                                Pole(node, same_location=True))
                        if hinge_w or hinge_u:
                            print(f'   -> am Knoten ({node.x}, {node.z}) '
                                  f'hat Stab {self.bars.index(bar)} '
                                  f'ein Normal-oder Querkraftgelenk')
                            print('     -> wird Relativpol')
                            direction = np.pi - bar.inclination if hinge_w \
                                else np.pi / 2 - bar.inclination
                            chain.add_relative_pole(
                                Pole(node, is_infinite=True,
                                     direction=direction))
                        else:
                            print(f'   -> am Knoten ({node.x}, {node.z}) '
                                  f'hat Stab {self.bars.index(bar)} '
                                  f'kein Gelenk')
                            print('     -> wird Absolutpol')
                            direction = np.pi / 2 - node.rotation \
                                if node.w == 'fixed' else np.pi - node.rotation
                            # bei overwrite = True gehts nicht! WARUM?!?!?
                            chain.set_absolute_pole(
                                Pole(node, direction=direction))
        # Starre Scheibe mit Gelenkprüfung
        #  -> kann durch Gelenke zum Absolutpol werden!
        elif node.u == 'fixed' and node.w == 'fixed' and node.phi == 'fixed':
            for bar in connected_bars:
                for end, hinge_phi, hinge_w, hinge_u in [
                    (bar.node_i,
                     bar.hinge_phi_i, bar.hinge_w_i, bar.hinge_u_i),
                    (bar.node_j,
                     bar.hinge_phi_j, bar.hinge_w_j, bar.hinge_u_j),
                ]:
                    if node == end:
                        if hinge_phi:
                            chain.set_absolute_pole(
                                Pole(node, same_location=True))
                        elif hinge_w or hinge_u:
                            direction = np.pi - bar.inclination if hinge_w \
                                else np.pi / 2 - bar.inclination
                            chain.set_absolute_pole(
                                Pole(node, is_infinite=True,
                                     direction=direction))
                        else:
                            chain.stiff = True
                            chain.set_absolute_pole(
                                Pole(node, same_location=True))

        # Wenn mindestens 2 Stäbe an den Knoten Anschließen
        # könnte es ein Verbindungsknoten sein:
        if len(connected_bars) > 1:
            # notwendig für zum finden eines Dreiecks
            # chain.add_connection_node(node)
            for bar in connected_bars:
                # if bar not in chain.bars:
                #     continue
                for end, hinge_phi, hinge_w, hinge_u in [
                    (bar.node_i,
                     bar.hinge_phi_i, bar.hinge_w_i, bar.hinge_u_i),
                    (bar.node_j,
                     bar.hinge_phi_j, bar.hinge_w_j, bar.hinge_u_j),
                ]:
                    if node == end:
                        # if hinge_w or hinge_u and hinge_phi:
                        #     print(f'   -> am Knoten ({node.x}, {node.z}) '
                        #           f'hat Stab {self.bars.index(bar)} '
                        #           f'ein Normal-oder Querkraftgelenk')
                        #     print('     -> wird Relativpol')
                        #     direction = np.pi - bar.inclination if hinge_w \
                        #         else np.pi / 2 - bar.inclination
                        #     chain.add_relative_pole(
                        #         Pole(node,is_infinite=True,
                        #              direction=direction))
                        if hinge_phi:
                            print(f'   -> am Knoten ({node.x}, {node.z}) '
                                  f'hat Stab {self.bars.index(bar)}'
                                  f' ein Momentengelenk')
                            print('     -> wird Relativpol')
                            chain.add_relative_pole(
                                Pole(node, same_location=True))
                        elif hinge_w or hinge_u:
                            print(f'   -> am Knoten ({node.x}, {node.z}) '
                                  f'hat Stab {self.bars.index(bar)} '
                                  f'ein Normal-oder Querkraftgelenk')
                            print('     -> wird Relativpol')
                            direction = np.pi - bar.inclination if hinge_w \
                                else np.pi / 2 - bar.inclination
                            chain.add_relative_pole(
                                Pole(node, is_infinite=True,
                                     direction=direction))

    def find_all_conn(self):
        conn = {}
        for chain in self.chains:
            for n in chain.connection_nodes:
                if n not in conn:
                    conn[n] = []
                conn[n].append(chain)

            for bar in chain.bars:
                for node in (bar.node_i, bar.node_j):
                    if node not in chain.connection_nodes and node in conn:
                        self._add_node_to_chain(chain, node)
        return conn

    def _identify_triangle(self):
        self.find_all_conn()
        self.print_chains()
        # Filter: Nur Scheiben mit >= 2 Verbindungsknoten
        chains = [chain for chain in self.chains if
                  len(chain.connection_nodes) >= 2]

        # Prüfen, ob mindestens 3 Scheiben vorhanden sind
        if len(chains) >= 3:
            # Alle Kombinationen von drei Scheiben prüfen
            for c1, c2, c3 in combinations(chains, 3):
                # Alle Verbindungsknoten sammeln
                # (mit Set zur Vermeidung von Duplikaten)
                nodes = set().union(
                    c1.connection_nodes,
                    c2.connection_nodes,
                    c3.connection_nodes)
                print(len(nodes))
                node_counts = {node: 0 for node in nodes}
                for chain in [c1, c2, c3]:
                    for node in chain.connection_nodes:
                        node_counts[node] += 1

                # Prüfen, ob genau 3 Knoten existieren,
                # die alle 2-mal vorkommen
                valid_triangle_nodes = [node for node, count in
                                        node_counts.items() if count == 2]

                # Prüfen, ob genau drei unterschiedliche Knoten existieren
                if len(valid_triangle_nodes) == 3:
                    print('-> Dreieck gefunden')
                    self._merge_chains([c1, c2, c3])
                    return True
        return False

    def _merge_chains(self, chains: list[Chain]):
        # Prüfe, ob alle zu mergen-Ketten in self.chains vorhanden sind
        missing_chains = [chain for chain in chains if
                          chain not in self.chains]
        if missing_chains:
            raise ValueError(
                f"Chain: {missing_chains} is not in self.chains.")

        # Sammle alle Bars aus den zu mergen-Ketten
        bars = set(bar for chain in chains for bar in chain.bars)

        # Entferne die Ketten direkt aus self.chains
        for chain in chains:
            self.chains.remove(chain)

        remaining_connection_nodes = set()
        for chain in chains:
            for node in chain.connection_nodes:
                remaining_connection_nodes.add(node)

        # Neue Dreiecksscheibe erstellen
        new_triangle_chain = Chain(bars)
        new_triangle_chain.add_connection_node(remaining_connection_nodes)

        for chain in chains:
            if chain.absolute_pole:
                new_triangle_chain.set_absolute_pole(chain.absolute_pole)

        for n in remaining_connection_nodes:
            self._add_node_to_chain(new_triangle_chain, n)

        self.chains.append(new_triangle_chain)
        self.print_chains()

    def _get_chain(self, bars: set[Bar]):
        for chain in self.chains:
            if bars & chain.bars:
                return chain
        return None

    def _get_chain_node(self, node):
        for chain in self.chains:
            for n in chain.connection_nodes:
                if node.same_location(n):
                    return chain
        return None

    def _find_shared_chains(self):
        """Finds all Chain objects that share at least one common bar."""
        bar_to_chains = {}

        for chain in self.chains:
            for bar in chain.bars:
                bar_to_chains.setdefault(bar, []).append(chain)

        # Sammle alle Chains, die mindestens ein gemeinsames Bar-Objekt haben
        return [
            chain for chains in bar_to_chains.values() if len(chains) > 1 for
            chain in chains
        ]

    # Pole finden
    @cached_property
    def solved(self):
        return (
            self._validation()
            and all(chain.solved is True for chain in self.chains)
        )

    @cached_property
    def node_to_chain_map(self):
        return {key: chains for key, chains in
                self.find_all_conn().items() if len(chains) > 1}

    def _identify_pole(self):
        for i, chain in enumerate(self.chains):
            print('------------------------------')
            print(f'///////// Scheibe {i} \\\\\\\\\\\\\\\\\\\\')
            print('Ist Scheibe vollständig?')
            if chain.solved:
                print(f'-> Scheibe {i} ist vollständig!')
                print(' -> sind angrenzende Scheiben unverschieblich?')
                break_outer_loop = False
                for rPole in list(chain.relative_pole):
                    if break_outer_loop:
                        break
                    connected_chain = (
                        self.node_to_chain_map.get(rPole.node, []))
                    if (len(connected_chain) > 1 and
                            any(c.stiff for c in connected_chain
                                if c != chain)):
                        print(' -> Ja, es gibt Starre Scheiben')
                        for conn_chain in connected_chain:
                            if conn_chain != chain and conn_chain.stiff:
                                self._set_aPole_connected_chain_is_stiff(
                                    chain, conn_chain, rPole)
                                break_outer_loop = True
                                break
                    else:
                        print(' -> Nein, es gibt keine Starren Scheiben')
                print('...Prüfung abgeschlossen!')
                continue

            print(f'-> Scheibe {i} ist unvollständig!')
            if not chain.solved_absolute_pole:
                print('  -> Absolutpolangaben sind nicht vollständig.')
                if chain.absolute_pole is None:
                    print('   -> Lage des Absolutpol unbekannt')
                    # Bildung von Absolut-Pollinien zweier angrenzenden
                    #  Scheiben (a) & (b)
                    # Im Schnittpunkt dieser Absolut-Pollinien liegt
                    #  der Absolutpol (i)
                    #   (a) - (a|i) -> (i)
                    #   (b) - (i|b) -> (i)
                    self._find_absolute_pole(chain)
                else:
                    print('   -> Koordinaten des Absolutpol unbekannt.')
                    print('    -> Absolutpol befindet sich auf bekannter '
                          'Pollinie')
                    # Scheibe wird durch ein bewegliches Lager gestützt:
                    #   -> der Absolutpol liegt auf einer Geraden senkrecht zur
                    #      möglichen Bewegungsrichtung des Lagers
                    absolute_pole_line = chain.absolute_pole.line()

                    # Bildung von Absolut-Pollinien einer angrenzenden Scheibe
                    # Im Schnittpunkt der beiden Geraden liegt der Absolutpol
                    self._find_absolute_pole(chain,
                                             [absolute_pole_line])

    def _find_absolute_pole(self, chain: Chain, lines: list = None):
        i = self.chains.index(chain)
        if lines is None:
            lines = []
        else:
            print(f'     -> g({i}): z(x) = {lines[0][0]} * x + {lines[0][1]}')
        # Finden der angrenzenden Scheiben und deren Pol-Linien
        print('Finden der angrenzenden Scheiben...')
        for rPole in chain.relative_pole:
            connected_chain = self.node_to_chain_map[rPole.node]
            intersection = set()
            print(f'Scheiben mit dem gleichen Verbindungsknoten '
                  f'({rPole.node.x} | {rPole.node.z}):')
            for conn_chain in connected_chain:
                if conn_chain != chain:
                    j = self.chains.index(conn_chain)
                    print('==================================================')
                    print(f' -> angrenzende Scheibe: {j}')
                    print(f'Ist Scheibe {j} eine starre Scheibe?')
                    if conn_chain.stiff:
                        self._set_aPole_connected_chain_is_stiff(
                            chain, conn_chain, rPole)
                        return True
                    print(f' -> Scheibe {j} ist keine starre Scheibe.')
                    print(f'Hat Scheibe {j} eine Absolutpollinie?')
                    if conn_chain.solved_absolute_pole:
                        line_dict = conn_chain.absolute_pole_lines_dict
                        if line_dict:
                            print(f' -> Absolutpollinie: ({j}) - ({i}|{j}):'
                                  f' z(x) = {line_dict[rPole.node][0]} * x '
                                  f'+ {line_dict[rPole.node][1]}')
                            lines.append(line_dict[rPole.node])
                else:
                    continue

                print('gespeicherte Geraden:')
                for nr, line in enumerate(lines):
                    print(f'  z{nr}(x) = {line[0]} * x + {line[1]}')

                if len(lines) == 2:
                    print('Haben die beiden Geraden eine Schnittpunkt?')
                    x, z = get_intersection_point(lines[0], lines[1])
                    print(f' -> Schnittpunkt: ({x},{z})')
                    intersection.add((x, z))
                    print(f'  -> Hinzufügen des Schnittpunktes zur '
                          f'Schnittpunktsliste: {intersection}')
                    lines.pop()
                    print('  -> letzte Gerade aus Geradenliste entfernen')
                else:
                    print(' -> aus einer Gerade kann kein Schnittpunkt '
                          'gebildet werden!')
            print('Wieviele Schnittpunkte liegen in der Schnittpunktliste?')
            if len(intersection) == 1:
                print(' -> ein schnittpunkt gefunden')
                x, z = intersection.pop()
                print(f'  -> Schnittpunkt ({x},{z}) als Absolutpol der Scheibe'
                      f' {i} hinzufügen.')
                if x is not None:
                    print('   -> Koordinaten des Absolutpols sind bekannt.')
                    if x == float('inf'):
                        aPole = chain.absolute_pole
                        x = aPole.node.x
                        z = aPole.node.z
                    chain.set_absolute_pole(
                        Pole(Node(x=x, z=z), same_location=True),
                        overwrite=True)
                elif x is None:
                    print('   -> Koordinaten des Absolutpols sind unbekannt.')
                    direction = lines[0][0]
                    if direction is None:
                        # für vertikale Linien muss None in 90°
                        #  umgewandelt werden!
                        direction = np.pi / 2
                    if chain.absolute_pole is None:
                        # Wenn für den Absolutpol noch keine Informationen
                        #  abgespeichert sind
                        p = next(iter(chain.relative_pole))
                        node = p.node
                    else:
                        # Wenn für den Absolutpol schon ein Knoten gespeichert
                        #  wurde
                        node = chain.absolute_pole.node
                    chain.set_absolute_pole(
                        Pole(node, is_infinite=True, direction=direction),
                        overwrite=True)
            elif len(intersection) > 1:
                print(' -> mehrere Schnittpunkte gefunden!!! \n'
                      '    -> Ein Absolutpol kann nur einen Koordinatenpunkt '
                      'haben \n'
                      '       -> Widerspruch im Polplan!')
            elif len(intersection) == 0:
                print(' -> keinen Schnittpunkt gefunden!')

    def _set_aPole_connected_chain_is_stiff(
            self, chain: Chain, conn_chain: Chain, rPole: Pole):
        j = self.chains.index(conn_chain)
        i = self.chains.index(chain)
        pole = None
        print(f' -> Scheibe {j} ist eine starre Scheibe!')
        aPole = chain.absolute_pole
        print(f' -> Hat Scheibe {i} einen Absolutpol?')
        print(f'    -> aPole: {aPole}')

        for rPole_conn_chain in conn_chain.relative_pole:
            print(f'  -> Ist der rPole ({i}|{j}) der Scheibe ({i}) '
                  f'gleich der Scheibe ({j})?')
            if rPole_conn_chain.node == rPole.node:
                if (not rPole_conn_chain.same_location
                        and rPole.same_location):
                    print(f'   -> Nein, rPole der Scheibe ({j}) '
                          f'liegt im Unendlichen')
                    pole = rPole_conn_chain
                    break
                elif (rPole_conn_chain.same_location
                      and not rPole.same_location):
                    print(f'   -> Nein, rPole der Scheibe ({i}) '
                          'liegt im Unendlichen')
                    pole = rPole
                    break
                else:
                    print('   -> Ja, sind identisch')
                    pole = rPole

        if pole.is_infinite:
            chain.set_absolute_pole(
                Pole(pole.node, direction=pole.direction, is_infinite=True),
                overwrite=False)
        else:
            print(f'  -> rPole ({i}|{j}) wird aPole ({i}) der Scheibe {i}')
            chain.set_absolute_pole(
                Pole(pole.node, same_location=True),
                overwrite=False)
        print(f'   -> rPole ({i}|{j}) wird aus Scheibe ({i}) entfernt')

        chain.relative_pole.remove(rPole)

        if len(chain.relative_pole) == 0:
            chain.add_relative_pole(Pole(pole.node, same_location=True))

    def _get_rPole_from_chain(self, node: Node, chain: Chain):
        for rPole in chain.relative_pole:
            if rPole.node == node:
                return rPole
        return None

    # Überprüfung
    def _validation(self):
        print('Validierung Start...')
        previous_chain = None
        for node, chains in self.node_to_chain_map.items():
            print('=========================')
            print('Knoten: ', node.x, node.z)
            print(f'Überprüfte Scheiben: '
                  f'{[self.chains.index(c) for c in chains]}')

            if previous_chain and previous_chain in chains:
                pairs = [(previous_chain, c) for c in chains if
                         c != previous_chain]
            else:
                pairs = combinations(chains, 2)

            for c1, c2 in pairs:
                if not self._validate_chain_pair(c1, c2, node):
                    return False

                previous_chain = c2
        return True

    def _validate_chain_pair(self, c1: Chain, c2: Chain, node: Node):
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)
        print('°°°°°°°°°°°°°°°°°°°°°°°°°°°')
        print(f'Kombo: {c1_idx} - {c2_idx}')
        if c1.stiff and not c2.stiff:
            print(f'  -> c{c1_idx} ist eine starre Scheibe')
            if c2.angle_factor == 0:
                c2.set_angle_factor(1)
                print(f'  -> c{c2_idx}.set_angle_factor: 1')
            return True

        if c2.stiff and not c1.stiff:
            print(f'  -> c{c2_idx} ist eine starre Scheibe')
            if c1.angle_factor == 0:
                c1.set_angle_factor(1)
                print(f'  -> c{c1_idx}.set_angle_factor: 1')
            return True

        if c1.stiff and c2.stiff:
            print(f'  -> c{c1_idx} & c{c2_idx} sind starre Scheiben')
            return True

        rPole = self._get_rPole_from_chain(node, c1)

        if not self._validation_lines(c1, c2, rPole):
            print('WIDERSPRUCH!!!')
            return False
        print(' -> Kein Konflikt')
        self._calc_angle_relation(c1, c2, rPole)
        return True

    def _validation_lines(self, c1: Chain, c2: Chain, rPole: Pole):
        print('----------------------------')
        print('# Pollinien werden überprüft')
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)

        line_1 = c1.absolute_pole_lines_dict[rPole.node]
        line_2 = c2.absolute_pole_lines_dict[rPole.node]

        print(f'  -> z{c1_idx} = {line_1[0]} * x + {line_1[1]}')
        print(f'  -> z{c2_idx} = {line_2[0]} * x + {line_2[1]}')

        x, z = get_intersection_point(line_1, line_2)

        if x == float('inf'):
            print('   -> identisch')
            return True
        elif x is None:
            print('   -> parallel')
            return True
        else:
            print('   -> Schnittpunkt')
            return False

    # Winkelbeziehung zwischen 2 Scheiben bestimmen
    def _calc_angle_relation(self, c1: Chain, c2: Chain, rPole: Pole):
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)
        print('----------------------------')
        print('# Winkelbeziehung aufstellen')
        print(f'     -> c{c2_idx}.angle = '
              f'c{c2_idx}.angle_factor * c{c1_idx}.angle')
        c1_distance = c1.vec_aPole_rPole_dict
        c2_distance = c2.vec_aPole_rPole_dict

        if c1_distance is None or c2_distance is None:
            print('    -> Ein Absolutpol liegt im Unendlichen \n'
                  '     --> Translation')
            if c1_distance is None:
                print(f'    -> aPol : ({c1_idx}) ist im Unendlichen')
                if c1.angle_factor == 0 or c1.angle_factor == 1:
                    print(f'    -> c{c1_idx}.angle_factor ist 0')
                    print(f'     -> c{c1_idx}.angle_factor wird auf 1 gesetzt')
                    c1.set_angle_factor(1)

                    # l1 = rPole.coords
                    a = np.array([[c1.absolute_pole.node.x],
                                  [c1.absolute_pole.node.z]])
                    l1 = rPole.coords - a

                    r21 = c2_distance[rPole]

                    # Längenverhältnis bestimmen und Richtung ermitteln
                    factor = (np.linalg.norm(l1) / np.linalg.norm(
                        r21)) * np.sign(np.dot(l1.T, r21)).item()

                    # Winkel-Faktor für c2 setzen
                    c2.set_angle_factor(factor)
                    print(f'c{c2_idx}.angle_factor = '
                          f'l{c1_idx} / r{c2_idx}{c1_idx}')
                    print(f'c{c2_idx}.angle_factor = '
                          f'(({c1_idx}) - ({c1_idx}|{c2_idx})) / (({c2_idx})'
                          f' - ({c1_idx}|{c2_idx}))')
                    print(f'c{c2_idx}.angle_factor = {factor}')
                else:
                    if c2.angle_factor == 0:
                        print(f'    -> c{c2_idx}.angle_factor ist 0')
                        print(f'     -> c{c2_idx}.angle_factor wird auf -1 '
                              f'gesetzt')
                        c2.set_angle_factor(1)
            if c2_distance is None:
                print(f'    -> aPol : ({c2_idx}) ist im Unendlichen')
                if c2.angle_factor == 0:
                    print(f'    -> c{c2_idx}.angle_factor ist 0')
                    print(f'     -> c{c2_idx}.angle_factor wird auf 1 gesetzt')
                    c2.set_angle_factor(1)

        elif rPole.is_infinite:
            # Wenn rPole = Querkraft- oder Normalkraftgelenk
            print('    -> rPole liegt im Unendlichen \n'
                  '     --> Translation')
            if c2.angle_factor == 0:
                print(f'    -> c{c2_idx}.angle_factor ist 0')
                print(f'     -> c{c2_idx}.angle_factor wird auf 1 gesetzt')
                c2.set_angle_factor(1)
        else:
            l1 = c1_distance[rPole]
            r21 = c2_distance[rPole]

            # Längenverhältnis bestimmen und Richtung ermitteln
            factor = (np.linalg.norm(l1) / np.linalg.norm(
                r21)) * np.sign(np.dot(l1.T, r21)).item()

            # Winkel-Faktor für c2 setzen
            c2.set_angle_factor(factor)
            print(f'c{c2_idx}.angle_factor = '
                  f'l{c1_idx} / r{c2_idx}{c1_idx}')
            print(f'c{c2_idx}.angle_factor = (({c1_idx}) - '
                  f'({c1_idx}|{c2_idx})) / (({c2_idx}) - ({c1_idx}|{c2_idx}))')
            print(f'c{c2_idx}.angle_factor = {factor}')

    # Bestimme alle Winkel
    def set_angle(self, target_chain, target_angle):
        print('---------------------------')
        print('Schritt 4: Winkelberechnung')
        print('---------------------------')
        target_chain_index = self.chains.index(target_chain)
        # Rückwärtsrechnen, um den Winkel für Scheibe 0 zu berechnen
        angle = target_angle

        chain_0 = self.chains[0]
        if chain_0 != target_chain:
            if not chain_0.stiff:
                chain_0.set_angle_factor(1)

        if target_chain.stiff:
            print(' -> Scheibe ist starr!')
            target_chain_index = target_chain_index + 1
            target_chain = self.chains[target_chain_index]
            if not target_chain.absolute_pole.is_infinite:
                angle = -1
            else:
                angle = 1
        target_chain.set_angle(angle)

        print(
            f'Berechnung des Winkel von Scheibe 0, so dass,\n'
            f'Scheibe {target_chain_index} den Winkel {angle} hat.')

        # Rückwärts iterieren bis Scheibe 0
        # TODO: das geht nur, wenn die Reihenfolge der Liste mit der der
        #  Geometrie übereinstimmt, wenn am Ende Scheiben zusammengefasst
        #  werden, ändert sich die Reihenfolge
        for i in range(target_chain_index - 1, -1, -1):
            print('i: ', i)
            next_chain = self.chains[i + 1]
            factor = next_chain.angle_factor

            print('factor: ', factor)

            if factor == 0:
                print(
                    f"Winkelberechnung für Kette {i + 1} abgebrochen: "
                    f"angle_factor = 0")
                break

            current_chain = self.chains[i]

            if current_chain.angle_factor == 0:
                print(
                    f"Winkelberechnung für Kette {i} abgebrochen:"
                    f" angle_factor = 0")
                break
            angle = angle / factor

            print(f"Berechneter Winkel für Scheibe {i}: {angle}")
            current_chain.set_angle(angle)

        print('(((((((((((((())))))))))))))')
        print('Berechnung aller Scheibenwinkel')
        # Vorwärts iterieren ab Scheibe target_chain_index
        # TODO: das funktioniert nur, wenn die Scheiben alle von anfang bis
        #  Ende durchnummeriert sind und nicht am Ende Dreiecke Identifiziert
        #  werden
        previous_chain = None
        for node, chains in self.node_to_chain_map.items():
            print('=========================')
            print('Knoten: ', node.x, node.z)
            print(f'Überprüfte Scheiben: '
                  f'{[self.chains.index(c) for c in chains]}')

            if previous_chain and previous_chain in chains:
                pairs = [(previous_chain, c) for c in chains if
                         c != previous_chain]
            else:
                pairs = combinations(chains, 2)

            for c1, c2 in pairs:
                self._calc_c2_angle_from_c1(c1, c2)
                previous_chain = c2

        print('-----------------------')
        print('Schritt 4 abgeschlossen')
        print('-----------------------')

    def _calc_c2_angle_from_c1(self, c1: Chain, c2: Chain):
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)
        print(f' -> Kombo: {c1_idx} - {c2_idx}')
        if c1.stiff or c2.stiff:
            print('stiff')
            return False

        angle = c1.angle * c2.angle_factor
        c2.set_angle(angle)
        print(
            f'   -> c{c2_idx}.angle = '
            f'c{c1_idx}.angle * c{c2_idx}.angle_factor')
        print(f'   -> c{c2_idx}.angle = {c1.angle} * {c2.angle_factor}')
        print(f'   -> c{c2_idx}.angle = {angle}')

    # Berechne Verschiebungsfigur
    def get_displacement_figure(self):
        displacement_bar_list: List[np.ndarray] = \
            [np.zeros((6, 1)) for _ in self.bars]
        bar_index_map = {bar: idx for idx, bar in enumerate(self.bars)}

        for i, chain in enumerate(self.chains):
            if chain.stiff:
                continue
            if chain.absolute_pole.is_infinite:
                displacement = self._calc_displacement_from_translation(chain)
                for bar in chain.bars:
                    idx = bar_index_map[bar]

                    displacement_bar = displacement_bar_list[idx]
                    displacement_bar[0:2, :] = displacement
                    displacement_bar[3:5, :] = displacement

                    displacement_bar = np.transpose(
                        bar.transformation_matrix()) @ displacement_bar

                    displacement_bar_list[idx] = displacement_bar

            else:
                center = chain.absolute_pole.coords
                angle = chain.angle
                for bar in chain.bars:
                    idx = bar_index_map[bar]

                    node_i = np.array([[bar.node_i.x], [bar.node_i.z]])
                    node_j = np.array([[bar.node_j.x], [bar.node_j.z]])

                    displacement_bar = displacement_bar_list[idx]
                    displacement_bar[0:2, :] = (
                        self._calc_displacement_from_rotation(
                            node_i, center, angle))
                    displacement_bar[3:5, :] = (
                        self._calc_displacement_from_rotation(
                            node_j, center, angle))

                    displacement_bar[2, :] = displacement_bar[5, :] = -angle

                    displacement_bar = np.transpose(
                        bar.transformation_matrix()) @ displacement_bar

                    displacement_bar_list[idx] = displacement_bar
        return displacement_bar_list

    def _calc_displacement_from_translation(self, chain: Chain):
        # bestimme die Richtung r basierend auf m
        m, _ = chain.absolute_pole.line()
        r = np.array([[1], [0]] if m is None else [[-m], [1]])
        # Normiere den Vektor
        v_norm = r / np.linalg.norm(r)
        # Iteriere über die relativen Pole der Scheibe
        for rPole in chain.relative_pole:
            # Iteriere über die verbundenen Scheiben
            for conn_chain in self.node_to_chain_map[rPole.node]:
                # Überprüft, ob es sich um eine verbundene Scheibe handelt
                if (conn_chain != chain and not conn_chain.stiff):
                    aPole_coords = np.array([
                        [conn_chain.absolute_pole.node.x],
                        [conn_chain.absolute_pole.node.z]])
                    rPole_coords = np.array([[rPole.node.x],
                                             [rPole.node.z]])
                    delta = aPole_coords - rPole_coords
                    r = np.hypot(delta[0][0], delta[1][0])

                    if np.sign(chain.angle) == np.sign(conn_chain.angle):
                        sign = 1
                    else:
                        sign = -1

                    return r * sign * conn_chain.angle * v_norm

    def _calc_displacement_from_rotation(self, point, center, angle):
        delta = point - center
        r = np.array([[0, -1], [1, 0]])
        return angle * r @ delta

    def _find_adjacent_chain(self, node, chain):
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

    # Print
    def print_chains(self):
        for chain in self.chains:
            i = self.chains.index(chain)
            index = []
            conn = []
            for bar in chain.bars:
                index.append(self.bars.index(bar))
            for n in chain.connection_nodes:
                conn.append((n.x, n.z))

            print(f'Chain: {i}, bars: {index}, \n -> conn_nodes: {conn}, '
                  f'\n -> rPol: {chain.relative_pole}, '
                  f'\n -> mPol: {chain.absolute_pole}'
                  f'\n -> starr: {chain.stiff}')


@dataclass(eq=False)
class SystemModifier:

    system: System

    def delete_load(self):
        bars = []
        for bar in self.system.bars:
            updated_bar = replace(
                bar,
                node_i=replace(bar.node_i, displacements=(), loads=()),
                node_j=replace(bar.node_j, displacements=(), loads=()),
                line_loads=(),
                point_loads=(),
                temp=BarTemp(temp_o=0, temp_u=0),
            )
            bars.append(updated_bar)
            return System(bars)

    def modify_bar_force(self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
                         position: float, virt_force: float = 1) -> System:
        for bar in self.system.bars:
            if bar == obj:
                # find list index of the bar
                bars = list(self.system.bars)
                idx = bars.index(bar)

                # set point_loads and hinge
                force_mapping = self.get_force_mapping()
                loads = force_mapping[force]['loads'](virt_force)
                hinge = force_mapping[force]['hinge']

                if 0 < position < 1:
                    # divide bar at position and get 2 Bar-Instances
                    bar_1, bar_2 = bar.segment([position])

                    # replace point_loads and hinge
                    bar_1 = replace(bar_1, hinge_phi_j=False, hinge_w_j=False,
                                    hinge_u_j=False)

                    modified_bar_1 = replace(bar_1, point_loads=loads[1],
                                             **{hinge[1]: True})
                    modified_bar_2 = replace(bar_2, point_loads=loads[0])

                    # replace the 2 modified bars in list
                    bars[idx:idx + 1] = [modified_bar_1, modified_bar_2]
                else:
                    # replace the modified bar in bars
                    bars[idx] = replace(bar, point_loads=loads[position],
                                        **{hinge[position]: True})

                    if position in (0, 1):
                        node = bar.node_i if position == 0 else bar.node_j
                        connected_bars = self.system.node_to_bar_map()[node]

                        if len(connected_bars) == 2:
                            for conn_bar in connected_bars:
                                if conn_bar != bar:
                                    index = bars.index(conn_bar)
                                    if conn_bar.node_i == node:
                                        bars[index] = replace(
                                            bars[index], point_loads=loads[0])
                                    else:
                                        bars[index] = replace(
                                            bars[index], point_loads=loads[1])
                return System(bars)
        raise ValueError("Bar not found in system")

    def modify_bar_deform(self, obj: Bar, deform: Literal['u', 'w', 'phi'],
                          position: float = 0) -> System:
        if deform not in ['u', 'w', 'phi']:
            raise ValueError(f"Invalid deform type: {deform}")
        for bar in self.system.bars:
            if bar == obj:
                # find list index of the bar
                bars = list(self.system.bars)
                idx = bars.index(bar)

                if position in (0, 1):
                    # set point_loads
                    force_mapping = self.get_deform_mapping()
                    loads = force_mapping[deform]

                    # replace the modified bar in bars
                    bars[idx] = replace(bar, point_loads=loads[1])

                    node = bar.node_i if position == 0 else bar.node_j
                    conn = self.system.node_to_bar_map()[node]

                    if len(conn) == 2:
                        bars[idx + 1] = replace(bars[idx + 1],
                                                point_loads=loads[0])
                else:
                    if deform == 'u':
                        point_load = BarPointLoad(1, 0, 0, 0, position)
                    elif deform == 'w':
                        point_load = BarPointLoad(0, 1, 0, 0, position)
                    else:
                        point_load = BarPointLoad(0, 0, 1, 0, position)
                    bars[idx] = replace(bar, point_loads=point_load)
                return System(bars)
        raise ValueError("Bar not found in system")

    def modify_node_force(self, obj: Node, force: Literal['fx', 'fz', 'fm'],
                          virt_force: float = 1):
        nodes = self.system.nodes()
        for node in nodes:
            if node == obj:
                if force == 'fx':
                    modified_node = (
                        replace(node,
                                u='free',
                                loads=[NodePointLoad(-virt_force, 0, 0)])
                    )
                elif force == 'fz':
                    modified_node = (
                        replace(node,
                                w='free',
                                loads=[NodePointLoad(0, -virt_force, 0)])
                    )
                elif force == 'fm':
                    modified_node = (
                        replace(node,
                                phi='free',
                                loads=[NodePointLoad(0, 0, -virt_force)])
                    )
                else:
                    raise ValueError(f"Invalid force type: {force}")

                connected_bars = self.system.node_to_bar_map()[node]
                bars = list(self.system.bars)

                for bar in connected_bars:
                    if node == bar.node_i:
                        modified_bar = replace(bar, node_i=modified_node)
                    else:
                        modified_bar = replace(bar, node_j=modified_node)

                    idx = bars.index(bar)
                    bars[idx] = modified_bar
                return System(bars)
        raise ValueError("Node not found in system")

    def modify_node_deform(self, obj: Node, deform: Literal['u', 'w', 'phi'],
                           virt_force: float = 1):
        nodes = self.system.nodes()
        for node in nodes:
            if node == obj:
                if deform == 'u':
                    modified_node = (
                        replace(node,
                                loads=[NodePointLoad(virt_force, 0, 0)])
                    )
                elif deform == 'w':
                    modified_node = (
                        replace(node,
                                loads=[NodePointLoad(0, virt_force, 0)])
                    )
                elif deform == 'phi':
                    modified_node = (
                        replace(node,
                                loads=[NodePointLoad(0, 0, -virt_force)])
                    )
                else:
                    raise ValueError(f"Invalid deform type: {deform}")

                connected_bars = self.system.node_to_bar_map()[node]
                bars = list(self.system.bars)

                for bar in connected_bars:
                    if node == bar.node_i:
                        modified_bar = replace(bar, node_i=modified_node)
                    else:
                        modified_bar = replace(bar, node_j=modified_node)

                    idx = bars.index(bar)
                    bars[idx] = modified_bar
                return System(bars)
        raise ValueError("Node not found in system")

    def get_force_mapping(self) -> Dict[str, Dict[str, Callable]]:
        return {
            'fx': {
                'loads': lambda virt_force: [
                    BarPointLoad(-virt_force, 0, 0, 0, 0),
                    BarPointLoad(virt_force, 0, 0, 0, 1)
                ],
                'hinge': ['hinge_u_i', 'hinge_u_j']
            },
            'fz': {
                'loads': lambda virt_force: [
                    BarPointLoad(0, -virt_force, 0, 0, 0),
                    BarPointLoad(0, virt_force, 0, 0, 1)
                ],
                'hinge': ['hinge_w_i', 'hinge_w_j']
            },
            'fm': {
                'loads': lambda virt_force: [
                    BarPointLoad(0, 0, -virt_force, 0, 0),
                    BarPointLoad(0, 0, virt_force, 0, 1)
                ],
                'hinge': ['hinge_phi_i', 'hinge_phi_j']
            }
        }

    def get_deform_mapping(self) -> Dict:
        return {
            'u': [
                    BarPointLoad(-1, 0, 0, 0, 0),
                    BarPointLoad(1, 0, 0, 0, 1)
                ],
            'w': [
                    BarPointLoad(0, -1, 0, 0, 0),
                    BarPointLoad(0, 1, 0, 0, 1)
                ],
            'phi': [
                    BarPointLoad(0, 0, -1, 0, 0),
                    BarPointLoad(0, 0, 1, 0, 1)
                ]
        }
