
from dataclasses import dataclass, replace, field
from functools import cache, cached_property
from typing import Literal, Dict, Callable
from itertools import combinations

import numpy as np

from sstatics.core import (
    Bar, BarTemp, BarPointLoad, Node, NodePointLoad
)


@dataclass(eq=False)
class Pol:

    node: Node
    same_location: bool = False
    bar: Bar = None

    def __post_init__(self):
        self.x = self.node.x
        self.z = self.node.z

    def __eq__(self, other):
        if isinstance(other, Pol):
            return self.x == other.x and self.z == other.z
        return False

    def __hash__(self):
        return hash((self.x, self.z))

    @property
    def points(self):
        if self.same_location:
            return self.x, self.z
        else:
            return None

    def line(self):
        print('hihi')
        if not self.same_location:
            if ((self.node.w == 'fixed' or self.node.u == 'fixed')
                    and self.node.phi == 'free'):
                if self.node.w == 'fixed':
                    if self.node.rotation:
                        m = np.tan(np.pi / 2 - self.node.rotation)
                        n = self.node.z - m * self.node.x
                    else:
                        # Vertikale Gerade
                        m, n = None, self.node.x
                else:  # node.u == 'fixed'
                    if self.node.rotation:
                        m = np.tan(np.pi - self.node.rotation)
                        n = self.node.z - m * self.node.x
                    else:
                        # Horizontale Gerade
                        m, n = 0, self.node.z
            else:
                raise ValueError(
                    "Die Knotenkonfiguration erlaubt keine eindeutige Gerade.")
            return m, n
        else:
            raise TypeError("The location of the Pol is in self.node.")


@dataclass(eq=False)
class Chain:

    bars: tuple[Bar, ...] | list[Bar]
    relative_pole: set = field(default_factory=set)
    absolue_pole: Pol = None
    connection_nodes: set = field(default_factory=set)
    stiff: bool = False

    def __post_init__(self):
        if len(self.bars) == 0:
            raise ValueError('There need to be at least one bar.')
        for i, bar in enumerate(self.bars[0:-1]):
            if any([
                bar.same_location(other_bar) for other_bar in self.bars[i + 1:]
            ]):
                raise ValueError(
                    'Cannot instantiate a chain with bars that share the '
                    'same location.'
                )

    @property
    def solved(self) -> bool:
        poles_valid = (
                self.solved_main_pole
                and self.solved_relative_pole
        )
        return (
                self.bars is not None
                and len(self.relative_pole) > 0
                and len(self.connection_nodes) > 0
                and poles_valid
        )

    @property
    def solved_main_pole(self) -> bool:
        return (
                self.absolue_pole is not None
                and self.absolue_pole.points is not None
        )

    @property
    def solved_relative_pole(self) -> bool:
        return all(pole.points is not None for pole in self.relative_pole)

    def add_connection_node(self, node: Node | set[Node]):
        if isinstance(node, Node):
            self.connection_nodes.add(node)
        elif isinstance(node, set):
            self.connection_nodes.update(node)
        else:
            raise TypeError("Expected a Node or a list of Nodes")

    def set_main_pole(self, pole: Pol):
        if not isinstance(pole, Pol):
            raise TypeError("absolue_pole must be an instance of Pol.")
        self.absolue_pole = pole

    def add_relative_pole(self, pole: Pol | list[Pol]):
        if isinstance(pole, Pol):
            self.relative_pole.add(pole)
        elif isinstance(pole, set):
            if not all(isinstance(p, Pol) for p in pole):
                raise TypeError(
                    "All items in the list must be instances of Pol.")
            self.relative_pole.update(pole)
        else:
            raise TypeError("Expected a Pol object or a list of Pol objects.")


@dataclass(eq=False)
class System:

    bars: tuple[Bar, ...] | list[Bar]

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

        self.segmented_bars = []
        for bar in self.bars:
            self.segmented_bars += bar.segment()
        self.segmented_bars = tuple(self.segmented_bars)

        self.dof = 3

    @cache
    def connected_nodes(self, segmented: bool = True):
        bars = self.segmented_bars if segmented else self.bars
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
        bars = self.segmented_bars if segmented else self.bars
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


@dataclass(eq=False)
class Polplan:

    system: System

    def __post_init__(self):
        self.bars = self.system.bars
        self.chains = []
        nodes = self.system.nodes(segmented=False)
        to_visit, visited = [nodes[0]], []
        while to_visit:
            current_node = to_visit.pop(0)
            if current_node not in visited:
                visited.append(current_node)
                to_visit += (
                    self.system.connected_nodes(segmented=False))[current_node]
                self._identify_chains(current_node)

        while self._identify_triangle():
            continue

        print(f'Polplan solved: {self.solved}')
        print(self.find_all_conn())
        self._identify_pole()

    # Scheiben finden
    def _identify_chains(self, current_node):
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(current_node.x, current_node.z)
        unassigned_bars = []
        connected_bars = (
            self.system.node_to_bar_map(segmented=False))[current_node]

        if len(connected_bars) == 1:
            self.new_chain(connected_bars[0], current_node)
        else:
            for bar in connected_bars:
                print('---------------------------------------------')
                print(f'Stabnr: {self.bars.index(bar)}')
                hinge_check = any([
                    bar.hinge_u_i, bar.hinge_w_i, bar.hinge_phi_i
                ]) if bar.node_i == current_node else any([
                    bar.hinge_u_j, bar.hinge_w_j, bar.hinge_phi_j
                ])
                if hinge_check:
                    print('hinge')
                    self.new_chain(bar, current_node)
                else:
                    print('nee nix hinge')
                    unassigned_bars.append(bar)
                    print(unassigned_bars)
            if unassigned_bars:
                print('noch unverarbeitete angrenzende Stäbe!')
                print(unassigned_bars)
                self.new_chain(unassigned_bars, current_node)

    def new_chain(self, bars, current_node):
        if not isinstance(bars, list):
            bars = [bars]
        existing_chain = self._get_chain(bars)
        index = []
        for bar in bars:
            index.append(self.bars.index(bar))
        print(index)
        if existing_chain:
            print(f'-> Stab {index} schon in Chain-nr.: '
                  f'{self.chains.index(existing_chain)}')
            self._add_bar_to_chain(existing_chain, bars)
            self._add_node_to_chain(existing_chain, current_node)
        else:
            print(f'-> Stab {index} noch nicht in Chains')
            new_chain = Chain(bars)
            self._add_node_to_chain(new_chain, current_node)
            self.chains.append(new_chain)
        print('Anzahl Chains: ', len(self.chains))
        self._identify_triangle()

    def _add_bar_to_chain(self, chain, bars):
        chain.bars.extend(
            bar for bar in bars if bar not in chain.bars)

    def _add_node_to_chain(self, chain, node):
        # Auflagerknoten
        if node.u == 'fixed' and node.w == 'fixed' and node.phi == 'free':
            # Knoten = Hauptpol
            chain.set_main_pole(Pol(node, same_location=True))
        elif (node.u == 'fixed' or node.w == 'fixed') and node.phi == 'free':
            # einwertiges Auflager -> senkrecht zur verhinderten Verschiebung
            chain.set_main_pole(Pol(node))
        elif node.u == 'fixed' and node.w == 'fixed' and node.phi == 'fixed':
            # Starre Scheibe
            # Pol zwischen Starren Scheibe und angrenzender Scheiben
            #     wird Hauptpol der angrenzenden Scheibe
            print('stiff')
            chain.stiff = True
            chain.set_main_pole(Pol(node, same_location=True))

        # potenzieller Verbindungspunkte zwischen weiteren Scheiben
        connected_bars = self.system.node_to_bar_map(segmented=False)[node]

        # Wenn mindestens 2 Stäbe an den Knoten Anschließen
        # könnte es ein Verbindungsknoten sein:
        if len(connected_bars) > 1:
            # notwendig für zum finden eines Dreiecks
            chain.add_connection_node(node)
            for bar in connected_bars:
                if bar.node_i == node:
                    if bar.hinge_phi_i:
                        chain.add_relative_pole(Pol(node, same_location=True))
                    elif bar.hinge_w_i or bar.hinge_u_i:
                        chain.add_relative_pole(Pol(node, bar=bar))
                elif bar.node_j == node:
                    if bar.hinge_phi_j:
                        chain.add_relative_pole(Pol(node, same_location=True))
                    elif bar.hinge_w_j or bar.hinge_u_j:
                        chain.add_relative_pole(Pol(node, bar=bar))

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
        bars = [bar for chain in chains for bar in chain.bars]

        # Entferne die Ketten direkt aus self.chains
        for chain in chains:
            self.chains.remove(chain)

        # Bestimme die verbleibenden Verbindungsknoten
        remaining_connection_nodes = {
            node
            for chain in chains
            for node in chain.connection_nodes
            if any(node in other_chain.connection_nodes for other_chain in
                   self.chains)
        }

        # Neue Dreiecksscheibe erstellen
        new_triangle_chain = Chain(bars)
        new_triangle_chain.add_connection_node(remaining_connection_nodes)

        for chain in chains:
            if chain.absolue_pole:
                new_triangle_chain.set_main_pole(chain.absolue_pole)

        for n in remaining_connection_nodes:
            self._add_node_to_chain(new_triangle_chain, n)

        self.chains.append(new_triangle_chain)
        self.print_chains()

    def _get_chain(self, bars):
        for chain in self.chains:
            if any(bar in chain.bars for bar in bars):
                return chain
        return None

    # Pole finden
    @property
    def solved(self):
        return (
            all(chain.solved is True for chain in self.chains)
        )

    @cached_property
    def node_to_chain_map(self):
        return self.find_all_conn()

    def _identify_pole(self):
        for chain in self.chains:
            if chain.solved:
                continue
            else:
                print('test')
                if not chain.solved_main_pole:
                    print('Hauptpol noch nicht gefunden!')

                if not chain.solved_relative_pole:
                    print('Relativ Pol noch nicht gefunden!')

    def berechne_schnittpunkt(self, m1, n1, m2, n2):
        """
        Berechnet den Schnittpunkt zweier Geraden, gegeben durch
        ihre Steigung (m) und den y-Achsenabschnitt (n).
        Berücksichtigt auch vertikale und horizontale Geraden,
        die durch spezielle Werte für m gekennzeichnet sind:
        - None für vertikale Geraden (unendliche Steigung).
        - 0.0 für horizontale Geraden.

        Args:
        m1, n1 : Steigung und y-Achsenabschnitt der ersten Geraden.
        m2, n2 : Steigung und y-Achsenabschnitt der zweiten Geraden.

        Returns:
        (x, y) : Koordinaten des Schnittpunkts der beiden Geraden.
        """

        # Fall: Beide Geraden sind vertikal
        if m1 is None and m2 is None:
            raise ValueError(
                "Beide Geraden sind vertikal und schneiden sich nicht.")

        # Fall: Eine Gerade ist vertikal
        if m1 is None:  # Erste Gerade ist vertikal
            x_schnitt = n1
            # x-Wert ist der x-Achsenabschnitt der ersten Geraden
            y_schnitt = m2 * x_schnitt + n2
            # Setze x in die Gleichung der zweiten Geraden
            return x_schnitt, y_schnitt

        if m2 is None:  # Zweite Gerade ist vertikal
            x_schnitt = n2
            # x-Wert ist der x-Achsenabschnitt der zweiten Geraden
            y_schnitt = m1 * x_schnitt + n1
            # Setze x in die Gleichung der ersten Geraden
            return x_schnitt, y_schnitt

        # Fall: Beide Geraden sind horizontal
        if m1 == 0.0 and m2 == 0.0:
            raise ValueError(
                "Beide Geraden sind horizontal und schneiden sich nicht.")

        # Fall: Eine Gerade ist horizontal
        if m1 == 0.0:  # Erste Gerade ist horizontal
            y_schnitt = n1
            # y-Wert ist der y-Achsenabschnitt der ersten Geraden
            x_schnitt = (y_schnitt - n2) / m2
            # Setze y in die Gleichung der zweiten Geraden
            return x_schnitt, y_schnitt

        if m2 == 0.0:  # Zweite Gerade ist horizontal
            y_schnitt = n2
            # y-Wert ist der y-Achsenabschnitt der zweiten Geraden
            x_schnitt = (y_schnitt - n1) / m1
            # Setze y in die Gleichung der ersten Geraden
            return x_schnitt, y_schnitt

        # Fall: Beide Geraden sind nicht parallel
        if m1 == m2:
            raise ValueError(
                "Die Geraden sind parallel oder identisch "
                "und haben keinen Schnittpunkt.")

        # Standardfall: Berechne den Schnittpunkt
        x_schnitt = (n2 - n1) / (m1 - m2)
        y_schnitt = m1 * x_schnitt + n1
        # Berechne y mit der ersten Geradengleichung

        return x_schnitt, y_schnitt

    def print_chains(self):
        for chain in self.chains:
            i = self.chains.index(chain)
            index = []
            conn = []
            for bar in chain.bars:
                index.append(self.bars.index(bar))
            for n in chain.connection_nodes:
                conn.append([n.x, n.z])

            print(f'Chain: {i}, bars: {index}, \n -> conn_nodes: {conn}, '
                  f'\n -> rPol: {chain.relative_pole}, '
                  f'\n -> mPol: {chain.absolue_pole}')


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
                    modified_bar_1 = replace(bar_1, point_loads=loads[1],
                                             **{hinge[1]: True})
                    modified_bar_2 = replace(bar_2, point_loads=loads[0])

                    # replace the 2 modified bars in list
                    bars[idx:idx + 1] = [modified_bar_1, modified_bar_2]
                else:
                    # replace point_loads and hinge
                    modified_bar = replace(bar, point_loads=loads[position],
                                           **{hinge[position]: True})
                    # replace the modified bar in list
                    bars[idx] = modified_bar

                return System(bars)
        raise ValueError("Bar not found in system")

    def modify_bar_deform(self, obj: Bar, deform: Literal['u', 'w', 'phi'],
                          position: float = 0) -> System:
        for bar in self.system.bars:
            if bar == obj:
                # set point_loads
                if deform == 'u':
                    point_loads = [BarPointLoad(1, 0, 0, 0, position)]
                elif deform == 'w':
                    point_loads = [BarPointLoad(0, 1, 0, 0, position)]
                elif deform == 'phi':
                    point_loads = [BarPointLoad(0, 0, -1, 0, position)]
                else:
                    raise ValueError(f"Invalid deform type: {deform}")

                # replace point_loads
                modified_bar = replace(bar, point_loads=point_loads)

                # replace the modified bar in bars
                # find list index of the bar
                bars = list(self.system.bars)
                idx = bars.index(bar)
                bars[idx] = modified_bar

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
