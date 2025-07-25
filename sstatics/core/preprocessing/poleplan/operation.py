
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set
from itertools import combinations

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.system import System

from sstatics.core.preprocessing.poleplan.objects import Chain, Pole


def get_intersection_point(line1, line2):
    """
    Calculate the intersection point of two lines.

    Args:
        line1 (tuple): (slope, intercept) of the first line.
        line2 (tuple): (slope, intercept) of the second line.

    Returns:
        tuple: (x, z) coordinates of the intersection point or
            (None, None) if no intersection.
    """
    m1, n1 = line1
    m2, n2 = line2

    # Check for identical lines
    if _are_lines_identical(m1, n1, m2, n2):
        print(u' -> z0 = z1 identische Geraden!')
        return float('inf'), float('inf')

    # Check for parallel lines
    if _are_lines_parallel(m1, m2):
        print(u' -> z0 und z1 sind parallele Geraden')
        return None, None

    # Handle vertical lines
    if m1 is None:
        return n1, m2 * n1 + n2
    if m2 is None:
        return n2, m1 * n2 + n1

    # Handle horizontal lines
    if m1 == 0:
        return (n1 - n2) / m2, n1
    if m2 == 0:
        return (n2 - n1) / m1, n2

    # General case: matrix representation
    matrix = np.array([[-m1, 1], [-m2, 1]])
    b = np.array([n1, n2])
    x, z = np.linalg.solve(matrix, b)
    return x, z


def _are_lines_identical(m1, n1, m2, n2, epsilon=1e-9):
    """
    Check if two lines are identical.

    Args:
        m1 (float): slope of the first line.
        n1 (float): intercept of the first line.
        m2 (float): slope of the second line.
        n2 (float): intercept of the second line.
        epsilon (float, optional): tolerance for floating-point
            comparison. Defaults to 1e-9.

    Returns:
        bool: True if the lines are identical, False otherwise.
    """
    return (np.isclose(m1, m2, atol=epsilon) and
            np.isclose(n1, n2, atol=epsilon))


def _are_lines_parallel(m1, m2, epsilon=1e-9):
    """
    Check if two lines are parallel.

    Args:
        m1 (float): slope of the first line.
        m2 (float): slope of the second line.
        epsilon (float, optional): tolerance for floating-point
            comparison. Defaults to 1e-9.

    Returns:
        bool: True if the lines are parallel, False otherwise.
    """
    return np.isclose(m1, m2, atol=epsilon)


def validate_point_on_line(line, point, debug=False, epsilon=1e-9):
    """
    Validate if a point lies on a line.

    Args:
        line (tuple): (slope, intercept) of the line.
        point (tuple): (x, z) coordinates of the point.
        debug (bool, optional): print debug information. Defaults to
            False.
        epsilon (float, optional): tolerance for floating-point
            comparison. Defaults to 1e-9.

    Returns:
        bool: True if the point lies on the line, False otherwise.
    """
    m, n = line
    x, z = point

    if m is None:  # Vertical line
        result = abs(x - n) < epsilon
    else:
        z_calc = m * x + n
        result = abs(z - z_calc) < epsilon

    if debug:
        status = u'JA' if result else u'NEIN'
        position = u'auf' if result else u'nicht auf'
        print(u'      -> {}, Punkt liegt {} der Geraden'.format(
            status, position))
    return result


def get_angle(point, center, displacement: float = 1):
    """
    Calculate the angle between a point and a center.

    Args:
        point (numpy array): coordinates of the point.
        center (numpy array): coordinates of the center.
        displacement (float, optional): displacement factor. Defaults
            to 1.

    Returns:
        float: angle between the point and the center.
    """
    r = point - center

    # Length of the vector
    length = np.linalg.norm(r)

    # Determine the sign
    if np.all(center == 0):
        sign = np.sign(r[0, 0])
    else:
        sign = np.sign(np.dot(r.T, center)).item()

    if displacement == 1:
        return sign / length
    else:
        return displacement / length

#############################################################################


@dataclass(eq=False)
class ChainIdentifier:
    system: System
    bars: List[Bar]
    chains: List[Chain]

    def run(self):
        nodes = self.system.nodes(segmented=False)
        to_visit, visited = [nodes[0]], []
        while to_visit:
            current_node = to_visit.pop(0)
            if current_node not in visited:
                visited.append(current_node)
                to_visit += (
                    self.system.connected_nodes(segmented=False))[current_node]
                self._identify_chains_from_node(current_node)

                self.print_chains()

                if shared_chains := self._find_shared_chains():
                    self._merge_chains(shared_chains)
                self._identify_triangle()

        while self._identify_triangle():
            continue

    def _identify_triangle(self):
        print('Dreieckssuche')
        self.find_all_conn()
        self.print_chains()
        chains = [chain for chain in self.chains if
                  len(chain.connection_nodes) >= 2]
        if len(chains) >= 3:
            for c1, c2, c3 in combinations(chains, 3):
                nodes = set().union(
                    c1.connection_nodes,
                    c2.connection_nodes,
                    c3.connection_nodes)
                node_counts = {node: 0 for node in nodes}
                for chain in [c1, c2, c3]:
                    for node in chain.connection_nodes:
                        node_counts[node] += 1
                valid_triangle_nodes = [node for node, count in
                                        node_counts.items() if count == 2]
                if len(valid_triangle_nodes) == 3:
                    print('--> gefunden!')
                    self._merge_chains([c1, c2, c3])
                    self.print_chains()
                    return True
        return False

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

    def _identify_chains_from_node(self, current_node: Node):
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f'({current_node.x},{current_node.z})')
        unassigned_bars = []
        connected_bars = (
            self.system.node_to_bar_map(segmented=False))[current_node]

        for bar in connected_bars:
            print('---------------------------------------------')
            print(f'Stabnr: {self.bars.index(bar)}')
            if self._has_hinge(bar, current_node):
                print(' -> Gelenk')
                print('   -> hier beginnt eine neue Scheibe')
                self._new_chain({bar}, current_node)
            else:
                print(' -> kein Gelenk')
                unassigned_bars.append(bar)

        print('Gibt es noch unverarbeitete angrenzende Stäbe?')
        if unassigned_bars:
            print(f' -> Ja, folgende Stäbe sind noch nicht zugewiesen: '
                  f'{[self.bars.index(b) for b in unassigned_bars]}')
            self._new_chain(set(unassigned_bars), current_node)

    def _new_chain(self, bars: Set[Bar], current_node: Node):
        existing_chain = self._get_chain(bars)
        if existing_chain:
            self._add_bar_to_chain(existing_chain, bars)
            self._add_node_to_chain(existing_chain, current_node)
        else:
            new_chain = Chain(bars)
            self._add_node_to_chain(new_chain, current_node)
            self.chains.append(new_chain)

    def _has_hinge(self, bar: Bar, node: Node):
        if bar.node_i == node:
            return any([bar.hinge_u_i, bar.hinge_w_i, bar.hinge_phi_i])
        else:
            return any([bar.hinge_u_j, bar.hinge_w_j, bar.hinge_phi_j])

    def _get_chain(self, bars: Set[Bar]):
        for chain in self.chains:
            if bars & chain.bars:
                return chain
        return None

    def _add_bar_to_chain(self, chain: Chain, bars: Set[Bar]):
        chain.add_bars(bars)

    def _add_node_to_chain(self, chain: Chain, node: Node):
        # TODO: das muss vereinfacht werden!
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

    def _find_shared_chains(self):
        bar_to_chains = {}
        for chain in self.chains:
            for bar in chain.bars:
                bar_to_chains.setdefault(bar, []).append(chain)
        return [
            chain for chains in bar_to_chains.values() if len(chains) > 1 for
            chain in chains
        ]

    def _merge_chains(self, chains: list[Chain]):
        # Prüfe, ob alle zu mergen-Ketten in self.chains vorhanden sind
        missing_chains = [chain for chain in chains if
                          chain not in self.chains]
        if missing_chains:
            raise ValueError(
                f"Chain: {missing_chains} is not in self.chains.")

        # Finde den kleinsten Index der vorkommenden Chains
        insertion_index = min(self.chains.index(chain) for chain in chains)

        # Sammle alle Bars aus den zu mergen-Ketten
        bars = set(bar for chain in chains for bar in chain.bars)

        # Entferne die Ketten direkt aus self.chains
        for chain in chains:
            print('Index: ')
            print(self.chains.index(chain))
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

        # Neue Chain an ursprünglicher Stelle einfügen
        self.chains.insert(insertion_index, new_triangle_chain)

        self.print_chains()

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

    def __call__(self):
        return self.run()


@dataclass(eq=False)
class PoleIdentifier:
    chains: List[Chain]
    node_to_chain_map: Dict[Node, List[Chain]]
    bars: List[Bar]

    # def identify_pole(self):
    #     for i, chain in enumerate(self.chains):
    #         if not chain.solved_absolute_pole:
    #             self._find_absolute_pole(chain)
    #
    # def _find_absolute_pole(self, chain: Chain, lines: List = None):
    #     i = self.chains.index(chain)
    #     if lines is None:
    #         lines = []
    #     for rPole in chain.relative_pole:
    #         connected_chain = self.node_to_chain_map.get(rPole.node, [])
    #         for conn_chain in connected_chain:
    #             if conn_chain != chain:
    #                 j = self.chains.index(conn_chain)
    #                 if conn_chain.solved_absolute_pole:
    #                     line_dict = conn_chain.absolute_pole_lines_dict
    #                     if line_dict:
    #                         lines.append(line_dict[rPole.node])
    #     if len(lines) == 2:
    #         x, z = get_intersection_point(lines[0], lines[1])
    #         if x is not None:
    #             chain.set_absolute_pole(
    #                 Pole(Node(x=x, z=z), same_location=True),
    #                 overwrite=True)

    def run(self):
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

    def __call__(self):
        return self.run()


@dataclass(eq=False)
class Validator:
    chains: List[Chain]
    node_to_chain_map: Dict[Node, List[Chain]]

    def run(self):
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

    def _get_rPole_from_chain(self, node: Node, chain: Chain):
        for rPole in chain.relative_pole:
            if rPole.node == node:
                return rPole
        return None

    # def validate(self):
    #     for node, chains in self.node_to_chain_map.items():
    #         for c1, c2 in combinations(chains, 2):
    #             if not self._validate_chain_pair(c1, c2, node):
    #                 return False
    #     return True
    #
    # def _validate_chain_pair(self, c1: Chain, c2: Chain, node: Node):
    #     rPole = self._get_rPole_from_chain(node, c1)
    #     if not self._validation_lines(c1, c2, rPole):
    #         return False
    #     self._calc_angle_relation(c1, c2, rPole)
    #     return True
    #
    # def _validation_lines(self, c1: Chain, c2: Chain, rPole: Pole):
    #     line_1 = c1.absolute_pole_lines_dict[rPole.node]
    #     line_2 = c2.absolute_pole_lines_dict[rPole.node]
    #     x, z = get_intersection_point(line_1, line_2)
    #     if x == float('inf'):
    #         return True
    #     elif x is None:
    #         return True
    #     else:
    #         return False
    #
    # def _calc_angle_relation(self, c1: Chain, c2: Chain, rPole: Pole):
    #     c1_idx = self.chains.index(c1)
    #     c2_idx = self.chains.index(c2)
    #     c1_distance = c1.vec_aPole_rPole_dict
    #     c2_distance = c2.vec_aPole_rPole_dict
    #     if c1_distance is None or c2_distance is None:
    #         if c1_distance is None:
    #             c1.set_angle_factor(1)
    #             factor = (np.linalg.norm(c1.absolute_pole.coords -
    #             rPole.coords) /
    #                       np.linalg.norm(c2_distance[rPole])) * np.sign(
    #                 np.dot((c1.absolute_pole.coords - rPole.coords).T,
    #                        c2_distance[rPole])).item()
    #             c2.set_angle_factor(factor)
    #         if c2_distance is None:
    #             c2.set_angle_factor(1)
    #
    #     elif rPole.is_infinite:
    #         c2.set_angle_factor(1)
    #     else:
    #         l1 = c1_distance[rPole]
    #         r21 = c2_distance[rPole]
    #         factor = (np.linalg.norm(l1) / np.linalg.norm(
    #             r21)) * np.sign(np.dot(l1.T, r21)).item()
    #         c2.set_angle_factor(factor)
    #
    # def _get_rPole_from_chain(self, node: Node, chain: Chain):
    #     for rPole in chain.relative_pole:
    #         if rPole.node == node:
    #             return rPole
    #     return None

    def __call__(self):
        return self.run()


@dataclass(eq=False)
class AngleCalculator:
    chains: List[Chain]
    node_to_chain_map: Dict[Node, List[Chain]]

    # def calculate_angle(self, target_chain, target_angle):
    #     target_chain_index = self.chains.index(target_chain)
    #     angle = target_angle
    #     target_chain.set_angle(angle)
    #     for i in range(target_chain_index - 1, -1, -1):
    #         next_chain = self.chains[i + 1]
    #         factor = next_chain.angle_factor
    #         current_chain = self.chains[i]
    #         angle = angle / factor
    #         current_chain.set_angle(angle)
    #     for node, chains in self.node_to_chain_map.items():
    #         for c1, c2 in combinations(chains, 2):
    #             self._calc_c2_angle_from_c1(c1, c2)
    #
    # def _calc_c2_angle_from_c1(self, c1: Chain, c2: Chain):
    #     c1_idx = self.chains.index(c1)
    #     c2_idx = self.chains.index(c2)
    #     if c1.stiff or c2.stiff:
    #         return False
    #     angle = c1.angle * c2.angle_factor
    #     c2.set_angle(angle)

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


@dataclass(eq=False)
class DisplacementCalculator:
    chains: List[Chain]
    bars: List[Bar]
    node_to_chain_map: Dict[Node, List[Chain]]

    # def calculate_displacement(self):
    #     displacement_bar_list = [np.zeros((6, 1)) for _ in self.bars]
    #     bar_index_map = {bar: idx for idx, bar in enumerate(self.bars)}
    #     for i, chain in enumerate(self.chains):
    #         if chain.stiff:
    #             continue
    #         if chain.absolute_pole.is_infinite:
    #             displacement = self._calc_displacement_from_translation(
    #             chain)
    #             for bar in chain.bars:
    #                 idx = bar_index_map[bar]
    #                 displacement_bar = displacement_bar_list[idx]
    #                 displacement_bar[0:2, :] = displacement
    #                 displacement_bar[3:5, :] = displacement
    #                 displacement_bar = np.transpose(
    #                     bar.transformation_matrix()) @ displacement_bar
    #                 displacement_bar_list[idx] = displacement_bar
    #         else:
    #             center = chain.absolute_pole.coords
    #             angle = chain.angle
    #             for bar in chain.bars:
    #                 idx = bar_index_map[bar]
    #                 node_i = np.array([[bar.node_i.x], [bar.node_i.z]])
    #                 node_j = np.array([[bar.node_j.x], [bar.node_j.z]])
    #                 displacement_bar = displacement_bar_list[idx]
    #                 displacement_bar[0:2, :] = (
    #                     self._calc_displacement_from_rotation(
    #                         node_i, center, angle))
    #                 displacement_bar[3:5, :] = (
    #                     self._calc_displacement_from_rotation(
    #                         node_j, center, angle))
    #                 displacement_bar[2, :] = displacement_bar[5, :] = -angle
    #                 displacement_bar = np.transpose(
    #                     bar.transformation_matrix()) @ displacement_bar
    #                 displacement_bar_list[idx] = displacement_bar
    #     return displacement_bar_list
    #
    # def _calc_displacement_from_translation(self, chain: Chain):
    #     m, _ = chain.absolute_pole.line()
    #     r = np.array([[1], [0]] if m is None else [[-m], [1]])
    #     v_norm = r / np.linalg.norm(r)
    #     for rPole in chain.relative_pole:
    #         for conn_chain in self.node_to_chain_map[rPole.node]:
    #             if conn_chain != chain and not conn_chain.stiff:
    #                 aPole_coords = np.array([
    #                     [conn_chain.absolute_pole.node.x],
    #                     [conn_chain.absolute_pole.node.z]])
    #                 rPole_coords = np.array([[rPole.node.x],
    #                                          [rPole.node.z]])
    #                 delta = aPole_coords - rPole_coords
    #                 r = np.hypot(delta[0][0], delta[1][0])
    #                 return r * np.sign(chain.angle) *
    #                 conn_chain.angle * v_norm
    #
    # def _calc_displacement_from_rotation(self, point, center, angle):
    #     delta = point - center
    #     r = np.array([[0, -1], [1, 0]])
    #     return angle * r @ delta

    def run(self):
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

    def __call__(self):
        return self.run()
