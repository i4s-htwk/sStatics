
from collections import defaultdict

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from itertools import combinations

from sstatics.core.logger_mixin import LoggerMixin
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

    is_identical, is_parallel = _check_lines(m1, n1, m2, n2)

    if is_identical:
        return float('inf'), float('inf')

    elif is_parallel:
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


def _check_lines(m1, n1, m2, n2, epsilon=1e-9):
    """
    Check if two lines are identical or parallel.

    Args:
        m1 (float or None): slope of the first line. None if vertical.
        n1 (float): intercept of the first line.
            If m1 is None, it's the x-coordinate.
        m2 (float or None): slope of the second line. None if vertical.
        n2 (float): intercept of the second line.
            If m2 is None, it's the x-coordinate.
        epsilon (float, optional): tolerance for floating-point comparison.
         Defaults to 1e-9.

    Returns:
        tuple: (is_identical, is_parallel)
    """
    if m1 is None and m2 is None:  # Both lines are vertical
        is_parallel = True
        is_identical = np.isclose(n1, n2, atol=epsilon)
    elif m1 is None or m2 is None:  # One line is vertical, the other is not
        is_parallel = False
        is_identical = False
    else:
        is_parallel = np.isclose(m1, m2, atol=epsilon)
        is_identical = is_parallel and np.isclose(n1, n2, atol=epsilon)
    return is_identical, is_parallel


def validate_point_on_line(line, point, debug=False, epsilon=1e-9):
    """
    Validate if a point lies on a line.

    Args:
        line (tuple): (slope, intercept) of the line.
        point (tuple): (x, z) coordinates of the point.
        debug (bool, optional): print debug information. Defaults to False.
        epsilon (float, optional): tolerance for floating-point \
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
        displacement (float, optional): displacement factor. Defaults to 1.

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


# help functions for logger messages
def dict_key_value_to_string(dictionary, all_chains) -> str:
    lines = []
    for node, chains in dictionary.items():
        lines.append(f"\nNode ({node.x}|{node.z})")
        lines.append("-" * 60)
        for chain in chains:
            i = all_chains.index(chain)
            lines.append(f"  Chain {i}:\n{chain}")
    return "\n".join(lines)


def chains_to_str(chains):
    lines = ["Identify chains:"]
    for i, chain in enumerate(chains):
        lines.append(f"Chain {i}:\n{chain}")
    return "\n".join(lines)


@dataclass(eq=False)
class ChainIdentifier(LoggerMixin):
    system: System
    debug: bool = False

    _node_to_chains: Optional[dict] = field(init=False, default=defaultdict)
    _chains: list[Chain] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.bars = self.system.bars
        self.logger.debug("ChainIdentifier post‑init completed")

    @property
    def node_to_chains(self):
        return self._node_to_chains

    @property
    def chains(self):
        return self._chains

    def run(self):
        """Main entry point – graph traversal and chain identification."""
        nodes = self.system.nodes(mesh_type='bars')
        to_visit, visited = [nodes[0]], []

        self.logger.info(
            f"Starting graph traversal with {len(nodes)} nodes.")
        self.logger.debug(
            f"Initial node to visit: ({nodes[0].x}, {nodes[0].z})")

        while to_visit:
            current_node = to_visit.pop(0)

            if current_node not in visited:
                self.logger.debug(
                    f"Visiting node ({current_node.x}, {current_node.z})"
                )
                visited.append(current_node)

                connected = (
                    self.system.connected_nodes(mesh_type='bars')
                )[current_node]
                self.logger.debug(
                    f"Connected nodes found: "
                    f"{[f'({n.x}, {n.z})' for n in connected]}"
                )

                # Add connected nodes to visit list
                to_visit += connected
                self.logger.debug(
                    f"Nodes to visit updated. Remaining: {len(to_visit)}"
                )

                # Identify new chains from the current node
                self.logger.info(
                    f"Identifying chains starting from node "
                    f"({current_node.x}, {current_node.z})"
                )
                self._identify_chains_from_node(current_node)
                self.logger.info(chains_to_str(self._chains))

                # Check for shared chains and merge if necessary
                if shared_chains := self._find_shared_chains():
                    self.logger.info(
                        f"Found {len(shared_chains)} shared chains — "
                        f"merging..."
                    )
                    self._merge_chains(shared_chains)
                    self.logger.debug("Shared chains merged successfully.")

                # Try identifying triangular connections
                self.logger.debug("Attempting to identify triangles.")
                self._identify_triangle()

        self.logger.info(
            f"Graph traversal completed. Visited {len(visited)} "
            f"nodes in total."
        )

        # Re‑run triangle detection until no further triangles are found.
        while self._identify_triangle():
            self.logger.debug("Another triangle was merged – re‑checking")
            continue

        self.logger.info("Chain identification finished")

    def _identify_triangle(self) -> bool:
        """Detect a closed triangle of three chains and merge them."""
        self.logger.info('Starting Identify Triangle.')
        self.find_all_conn()
        chains = [chain for chain in self._chains if
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
                    self.logger.debug(
                        "Valid triangle found between chains:\n"
                        + chains_to_str([c1, c2, c3])
                    )
                    self._merge_chains([c1, c2, c3])
                    self.logger.info("Triangle merged into a new chain")
                    return True
        self.logger.debug("No triangle found in this iteration")
        return False

    def find_all_conn(self):
        """Populate the node‑to‑chains dictionary for the current chain set."""
        node_to_chains: dict[Node, List[Chain]] = {}
        for chain in self._chains:
            for node in chain.connection_nodes:
                node_to_chains.setdefault(node, []).append(chain)
            for bar in chain.bars:
                for node in (bar.node_i, bar.node_j):
                    if (node not in chain.connection_nodes and
                            node in node_to_chains):
                        self._add_node_to_chain(chain, node)
        self._node_to_chains = node_to_chains
        self.logger.debug(
            "Node‑to‑chains mapping created:\n"
            + dict_key_value_to_string(node_to_chains, self._chains)
        )

    def _identify_chains_from_node(self, current_node: Node):
        """Create new chains starting at the supplied node."""
        unassigned_bars = []
        connected_bars = (
            self.system.node_to_bar_map(segmented=False))[current_node]
        self.logger.debug(
            "Processing %d bars connected to node (%s, %s)",
            len(connected_bars), current_node.x, current_node.z,
        )
        self.logger.debug('Iteration over all bars that are connected with '
                          'this node')
        for bar in connected_bars:
            bar_idx = self.bars.index(bar)
            self.logger.debug(f"Connected Bar: {bar_idx}")

            if self._has_hinge(bar, current_node):
                self.logger.debug("Hinge detected – starting a new chain")
                self._new_chain({bar}, current_node)
            else:
                self.logger.debug("No hinge – bar will be stored for later")
                unassigned_bars.append(bar)

        self.logger.debug("Checking for unprocessed adjacent bars")
        if unassigned_bars:
            indices = [self.bars.index(b) for b in unassigned_bars]
            self.logger.debug(
                "Unassigned bars remain: %s", ", ".join(map(str, indices))
            )
            self._new_chain(set(unassigned_bars), current_node)
        else:
            self.logger.debug("All connected bars have been assigned")

    def _new_chain(self, bars: Set[Bar], current_node: Node):
        """Create a new chain or extend an existing one."""
        existing = self._get_chain(bars)
        if existing:
            self.logger.debug(
                "Extending existing chain (now %d bars)", len(existing.bars)
            )
            existing.add_bars(bars)
            self._add_node_to_chain(existing, current_node)
        else:
            self.logger.info(
                "Creating new chain with %d bars at node (%s, %s)",
                len(bars), current_node.x, current_node.z,
            )
            new_chain = Chain(bars)
            self._add_node_to_chain(new_chain, current_node)
            self._chains.append(new_chain)

    @staticmethod
    def _has_hinge(bar: Bar, node: Node):
        """Return True if the bar has a hinge at the given node."""
        if bar.node_i == node:
            return any(bar.hinge[0:3])
        else:
            return any(bar.hinge[4:6])

    def _get_chain(self, bars: Set[Bar]):
        """Find a chain that already contains any of the supplied bars."""
        for chain in self._chains:
            if bars & chain.bars:
                return chain
        return None

    def _add_node_to_chain(self, chain: Chain, node: Node):
        # TODO: simplify
        """Add a node (and possibly poles) to a chain."""
        self.logger.debug(
            "Adding node (%s, %s) to chain (currently %d nodes)",
            node.x, node.z, len(chain.connection_nodes),
        )

        connected_bars = self.system.node_to_bar_map(segmented=False)[node]
        chain.add_connection_node(node)

        # Determine pole type based on support conditions
        if node.u == "fixed" and node.w == "fixed" and node.phi == "free":
            self.logger.info(
                "Node is a two‑degree‑of‑freedom support – absolute pole added"
            )
            chain.set_absolute_pole(Pole(node, same_location=True))

        elif (node.u == "fixed" or node.w == "fixed") and node.phi == "free":
            self.logger.info(
                "Node is a one‑degree‑of‑freedom support – checking attached"
                "bars"
            )
            for bar in connected_bars:
                for end, hinge_phi, hinge_w, hinge_u in \
                        [
                            (bar.node_i,
                             bar.hinge_phi_i, bar.hinge_w_i, bar.hinge_u_i
                             ),
                            (bar.node_j,
                             bar.hinge_phi_j, bar.hinge_w_j, bar.hinge_u_j
                             )
                        ]:
                    if node != end:
                        continue

                    if hinge_phi:
                        self.logger.debug(
                            "Moment hinge found on bar %d – adding relative "
                            "pole",
                            self.bars.index(bar),
                        )
                        chain.add_relative_pole(Pole(node, same_location=True))

                    if hinge_w or hinge_u:
                        direction = (
                            np.pi - bar.inclination
                            if hinge_w
                            else np.pi / 2 - bar.inclination
                        )
                        self.logger.debug(
                            "Normal/shear hinge on bar %d – adding infinite "
                            "relative pole",
                            self.bars.index(bar),
                        )
                        chain.add_relative_pole(
                            Pole(node, is_infinite=True, direction=direction)
                        )
                    else:
                        direction = (
                            np.pi / 2 - node.rotation
                            if node.w == "fixed"
                            else np.pi - node.rotation
                        )
                        self.logger.debug(
                            "No hinge – setting absolute pole with "
                            "direction %.3f",
                            direction,
                        )
                        chain.set_absolute_pole(
                            Pole(node, direction=direction))

        elif node.u == "fixed" and node.w == "fixed" and node.phi == "fixed":
            self.logger.info(
                "Fully fixed node – checking for stiff or pole conditions")
            for bar in connected_bars:
                for end, hinge_phi, hinge_w, hinge_u in \
                        [
                            (bar.node_i,
                             bar.hinge_phi_i, bar.hinge_w_i, bar.hinge_u_i
                             ),
                            (bar.node_j,
                             bar.hinge_phi_j, bar.hinge_w_j, bar.hinge_u_j
                             )
                        ]:
                    if node != end:
                        continue

                    if hinge_phi:
                        chain.set_absolute_pole(Pole(node, same_location=True))
                    elif hinge_w or hinge_u:
                        direction = (
                            np.pi - bar.inclination
                            if hinge_w
                            else np.pi / 2 - bar.inclination
                        )
                        chain.set_absolute_pole(
                            Pole(node, is_infinite=True, direction=direction)
                        )
                    else:
                        chain.stiff = True
                        chain.set_absolute_pole(Pole(node, same_location=True))

        # If more than one bar meets at the node, it may become a connection
        # node
        if len(connected_bars) > 1:
            for bar in connected_bars:
                for end, hinge_phi, hinge_w, hinge_u in \
                        [
                            (bar.node_i,
                             bar.hinge_phi_i, bar.hinge_w_i, bar.hinge_u_i
                             ),
                            (bar.node_j,
                             bar.hinge_phi_j, bar.hinge_w_j, bar.hinge_u_j
                             )
                        ]:
                    if node != end:
                        continue

                    if hinge_phi:
                        self.logger.debug(
                            "Moment hinge on bar %d at connection node – "
                            "relative pole",
                            self.bars.index(bar),
                        )
                        chain.add_relative_pole(Pole(node, same_location=True))
                    elif hinge_w or hinge_u:
                        direction = (
                            np.pi - bar.inclination
                            if hinge_w
                            else np.pi / 2 - bar.inclination
                        )
                        self.logger.debug(
                            "Normal/shear hinge on bar %d at connection node "
                            "– infinite relative pole",
                            self.bars.index(bar),
                        )
                        chain.add_relative_pole(
                            Pole(node, is_infinite=True, direction=direction)
                        )

    def _find_shared_chains(self):
        """Return a flat list of chains that share at least one bar."""
        bar_to_chains = {}
        for chain in self._chains:
            for bar in chain.bars:
                bar_to_chains.setdefault(bar, []).append(chain)

        shared = [
            chain
            for chains in bar_to_chains.values()
            if len(chains) > 1
            for chain in chains
        ]
        self.logger.debug(
            "Shared chains detection yielded %d entries", len(shared)
        )
        return shared

    def _merge_chains(self, chains: List[Chain]):
        """Merge the supplied chains into a single new chain."""
        missing = [c for c in chains if c not in self._chains]
        if missing:
            msg = f"Chain(s) {missing} not present in self._chains"
            self.logger.error(msg)
            raise ValueError(msg)

        insertion_index = min(self._chains.index(c) for c in chains)
        self.logger.debug(
            "Merging %d chains at insertion index %d", len(chains),
            insertion_index
        )

        combined_bars = {bar for chain in chains for bar in chain.bars}
        self.logger.debug("Combined bar count: %d", len(combined_bars))

        # Remove the old chains
        for chain in chains:
            self.logger.debug("Removing chain with %d bars", len(chain.bars))
            self._chains.remove(chain)

        # Gather all distinct connection nodes
        remaining_nodes = {node for chain in chains for node in
                           chain.connection_nodes}
        self.logger.debug("Remaining connection nodes count: %d",
                          len(remaining_nodes))

        # Build the new chain
        new_chain = Chain(combined_bars)
        new_chain.add_connection_node(remaining_nodes)
        self.logger.info(
            "Creating new chain with %d bars",
            len(combined_bars)
        )

        for chain in chains:
            if chain.absolute_pole:
                new_chain.set_absolute_pole(chain.absolute_pole)

        for node in remaining_nodes:
            self._add_node_to_chain(new_chain, node)

        self._chains.insert(insertion_index, new_chain)
        self.logger.info(
            "Merged %d chains into a new chain (now %d total chains) \n"
            "new chain:\n%s",
            len(chains), len(self._chains), new_chain
        )

    def __call__(self):
        """Convenient entry point so the class can be called like a
        function."""
        self.logger.debug("ChainIdentifier invoked via __call__")
        return self.run()


@dataclass(eq=False)
class PoleIdentifier(LoggerMixin):
    """
    Analyze kinematic chains and determine missing absolute poles.

    Traverses all chains, identifies missing absolute poles, and infers
    their positions from geometric or kinematic relationships between
    adjacent chains.

    Attributes
    ----------
    chains : List[Chain]
        List of chain objects to be analyzed.
    node_to_chains : Dict[Node, List[Chain]]
        Mapping from each node to the chains connected to it.
    debug : bool, optional
        Enables verbose debugging if True (default: False).
    """

    chains: List[Chain]
    node_to_chains: Dict[Node, List[Chain]]
    debug: bool = False

    def run(self):
        """
        Iterate over all chains and resolve absolute poles.

        Checks each chain for missing absolute pole information and infers
        absolute poles by analyzing relationships between adjacent chains.
        """
        self.logger.info(
            "Pole identification started – %d chains to process",
            len(self.chains)
        )

        for i, chain in enumerate(self.chains):
            self.logger.info("Processing chain %d", i)

            if chain.solved:
                self.logger.info("Chain %d is already solved", i)
                self._check_adjacent_stiff_chains(i, chain)
                self.logger.debug("Verification for chain %d completed", i)
                continue

            self.logger.debug("Chain %d is incomplete", i)

            if not chain.solved_absolute_pole:
                self.logger.debug(
                    "Absolute pole data missing for chain %d", i
                )

                if chain.absolute_pole is None:
                    self.logger.info(
                        "Absolute pole unknown – attempting inference from "
                        "connected chains"
                    )
                    self._find_absolute_pole(chain)
                else:
                    self.logger.info(
                        "Absolute pole exists but coordinates unknown"
                    )
                    abs_line = chain.absolute_pole.line()
                    self._find_absolute_pole(chain, [abs_line])

    def _check_adjacent_stiff_chains(self, i: int, chain: Chain):
        """
        Check whether any adjacent chain connected to the given chain is stiff.
        """
        self.logger.debug("Checking adjacent stiff chains for chain %s", i)

        for r_pole in list(chain.relative_pole):
            connected = self.node_to_chains.get(r_pole.node, [])
            if (len(connected) > 1 and
                    any(c.stiff for c in connected if c != chain)):
                self.logger.info(
                    "Stiff adjacent chain detected for chain %s "
                    "via node(%s, %s)",
                    i, r_pole.node.x, r_pole.node.z
                )
                for adj in connected:
                    if adj != chain and adj.stiff:
                        self._set_aPole_connected_chain_is_stiff(
                            chain, adj, r_pole)
                        return
            else:
                self.logger.debug(
                    "No stiff adjacent chains found at node (%s, %s)",
                    r_pole.node.x, r_pole.node.z
                )

    def _find_absolute_pole(self, chain: Chain, lines: list = None):
        """
        Infer the absolute pole of a chain by analyzing intersections of
        absolute pole lines.
        """
        i = self.chains.index(chain)
        if lines is None:
            lines = []
        else:
            self.logger.debug(
                "Chain %s: Using existing line equation z(x) = %s * x + %s",
                i, lines[0][0], lines[0][1]
            )

        self.logger.debug("Searching for adjacent chains for chain %s", i)

        intersection = set()
        for rPole in chain.relative_pole:
            connected_chain = self.node_to_chains[rPole.node]
            self.logger.debug(
                "Node (%s | %s) connects %s chains",
                rPole.node.x, rPole.node.z, len(connected_chain)
            )

            for conn_chain in connected_chain:
                if conn_chain == chain:
                    continue

                j = self.chains.index(conn_chain)
                self.logger.debug(
                    "Evaluating adjacent chain %s for chain %s", j, i
                )

                if conn_chain.stiff:
                    self.logger.info(
                        "Adjacent chain %s is stiff – using for absolute pole "
                        "identification", j
                    )
                    self._set_aPole_connected_chain_is_stiff(
                        chain, conn_chain, rPole
                    )
                    return True

                if conn_chain.solved_absolute_pole:
                    line_dict = conn_chain.apole_lines
                    if line_dict:
                        line = line_dict[rPole.node]
                        self.logger.debug(
                            "Adding absolute pole line from chain %s: "
                            "z(x) = %s * x + %s",
                            j, line[0], line[1]
                        )
                        lines.append(line)

                if len(lines) == 2:
                    x, z = get_intersection_point(lines[0], lines[1])
                    intersection.add((x, z))
                    self.logger.debug("Intersection found: (%s, %s)", x, z)
                    lines.pop()
                else:
                    self.logger.debug(
                        "Not enough lines to form an intersection "
                        "(need 2, have %s)",
                        len(lines)
                    )

        if len(intersection) == 1:
            x, z = intersection.pop()
            self.logger.info(
                "Single intersection found for chain %s at (%s, %s)", i, x, z
            )
            if x is not None:
                if x == float('inf'):
                    aPole = chain.absolute_pole
                    x, z = aPole.node.x, aPole.node.z
                chain.set_absolute_pole(
                    Pole(Node(x=x, z=z), same_location=True), overwrite=True
                )
            else:
                direction = lines[0][0] or np.pi / 2
                node = chain.absolute_pole.node if chain.absolute_pole \
                    else next(iter(chain.relative_pole)).node
                chain.set_absolute_pole(
                    Pole(node, is_infinite=True, direction=direction),
                    overwrite=True
                )
        elif len(intersection) > 1:
            self.logger.warning(
                "Multiple intersection points found for chain %s – "
                "inconsistent pole geometry",
                i
            )
        else:
            self.logger.debug("No intersection found for chain %s", i)

    def _set_aPole_connected_chain_is_stiff(
            self, chain: Chain, conn_chain: Chain, rPole: Pole
    ):
        """
        Assign the absolute pole of a chain if a connected chain is stiff.
        """
        j = self.chains.index(conn_chain)
        i = self.chains.index(chain)
        pole = None

        self.logger.info(
            "Connected chain %s is stiff – setting absolute pole for chain %s",
            j, i
        )

        for rPole_conn_chain in conn_chain.relative_pole:
            if rPole_conn_chain.node == rPole.node:
                if not rPole_conn_chain.same_location and rPole.same_location:
                    self.logger.debug(
                        "Relative pole (%s|%s) of connected chain lies at "
                        "infinity", i, j
                    )
                    pole = rPole_conn_chain
                    break
                elif (rPole_conn_chain.same_location and
                      not rPole.same_location):
                    self.logger.debug(
                        "Relative pole (%s|%s) of current chain lies at "
                        "infinity", i, j
                    )
                    pole = rPole
                    break
                else:
                    pole = rPole

        if pole.is_infinite:
            chain.set_absolute_pole(
                Pole(pole.node, direction=pole.direction, is_infinite=True),
                overwrite=False
            )
        else:
            chain.set_absolute_pole(
                Pole(pole.node, same_location=True), overwrite=False
            )

        chain.relative_pole.remove(rPole)

        if len(chain.relative_pole) == 0:
            chain.add_relative_pole(Pole(pole.node, same_location=True))

        self.logger.debug(
            "Relative pole (%s|%s) removed and absolute pole assigned for "
            "chain %s",
            i, j, i
        )

    def __call__(self):
        """
        Run pole identification directly via callable instance syntax.
        """
        self.logger.debug("PoleIdentifier invoked via __call__()")
        return self.run()


@dataclass(eq=False)
class Validator(LoggerMixin):
    """
    Validate geometric and stiffness relationships between chains.

    Parameters
    ----------
    chains : List[Chain]
        List of all chains in the model.
    node_to_chains : Dict[Node, List[Chain]]
        Mapping from each node to the chains that contain it.
    debug : bool, optional
        If ``True``, additional debug information is logged. Default is
        ``False``.

    Attributes
    ----------
    _solved : bool
        Internal flag indicating the result of the last validation run.
    """

    chains: List[Chain]
    node_to_chains: Dict[Node, List[Chain]]
    debug: bool = False

    _solved: Optional[bool] = field(init=False, default=False)

    @property
    def solved(self) -> bool:
        """Return the result of the validation."""
        return self._solved

    def run(self) -> bool:
        """
        Execute the full validation routine.

        Returns
        -------
        bool
            ``True`` if all chain pairs are consistent, ``False`` otherwise.
        """
        self.logger.info("Validation started")
        previous_chain = None

        for node, chains in self.node_to_chains.items():
            self.logger.info(
                "Processing node at (x=%s, z=%s)", node.x, node.z
            )
            chain_indices = [self.chains.index(c) for c in chains]
            self.logger.debug("Chains attached to node: %s", chain_indices)

            if previous_chain and previous_chain in chains:
                pairs = [
                    (previous_chain, c) for c in chains if c != previous_chain
                ]
                self.logger.debug(
                    "Using previous chain %s as first element in pairwise "
                    "checks",
                    self.chains.index(previous_chain),
                )
            else:
                pairs = list(combinations(chains, 2))
                self.logger.debug(
                    "Generating %d pairwise combinations", len(pairs)
                )

            for c1, c2 in pairs:
                if not self._validate_chain_pair(c1, c2, node):
                    self._solved = False
                    self.logger.error("Validation failed – aborting")
                    return False
                previous_chain = c2

        self._solved = True
        self.logger.info("Validation completed successfully")
        return True

    def _validate_chain_pair(self, c1: Chain, c2: Chain, node: Node) -> bool:
        """
        Validate a pair of chains that share a common node.

        Parameters
        ----------
        c1, c2 : Chain
            The two to be compared.
        node : Node
            The node shared by the two chains.

        Returns
        -------
        bool
            ``True`` if the pair is consistent, ``False`` otherwise.
        """
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)

        self.logger.debug("Validating chain pair (%d, %d)", c1_idx, c2_idx)

        # Stiff‑chain handling
        if c1.stiff and not c2.stiff:
            self.logger.info("Chain %d is stiff; adjusting chain %d",
                             c1_idx, c2_idx)
            if c2.angle_factor == 0:
                c2.angle_factor = 1
                self.logger.debug("Set angle_factor of chain %d to 1",
                                  c2_idx)
            return True

        if c2.stiff and not c1.stiff:
            self.logger.info("Chain %d is stiff; adjusting chain %d",
                             c2_idx, c1_idx)
            if c1.angle_factor == 0:
                c1.angle_factor = 1
                self.logger.debug("Set angle_factor of chain %d to 1",
                                  c1_idx)
            return True

        if c1.stiff and c2.stiff:
            self.logger.info(
                "Both chains %d and %d are stiff – no conflict",
                c1_idx, c2_idx
            )
            return True

        # General geometric validation
        r_pole = self._get_rPole_from_chain(node, c1)
        if not self._validation_lines(c1, c2, r_pole):
            self.logger.warning(
                "Geometric contradiction detected between chains %d and %d "
                "at node (x=%s, z=%s)",
                c1_idx,
                c2_idx,
                node.x,
                node.z,
            )
            return False

        self.logger.info(
            "No contradiction for chain pair (%d, %d); computing angle "
            "relation",
            c1_idx,
            c2_idx,
        )
        self._calc_angle_relation(c1, c2, r_pole)
        return True

    def _validation_lines(self, c1: Chain, c2: Chain, r_pole: Pole) -> bool:
        """
        Compare the absolute‑pole lines of two chains at a given relative pole.

        Parameters
        ----------
        c1, c2 : Chain
            Chains whose lines are compared.
        r_pole : Pole
            The relative pole common to both chains.

        Returns
        -------
        bool
            ``True`` if the lines are compatible (identical or parallel),
            ``False`` if they intersect (indicating a conflict).
        """
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)

        line_1 = c1.apole_lines[r_pole.node]
        line_2 = c2.apole_lines[r_pole.node]

        self.logger.debug(
            "Chain %d line: z = %s * x + %s", c1_idx, line_1[0], line_1[1]
        )
        self.logger.debug(
            "Chain %d line: z = %s * x + %s", c2_idx, line_2[0], line_2[1]
        )

        x, z = get_intersection_point(line_1, line_2)

        if x == float("inf"):
            self.logger.debug("Lines are identical – no conflict")
            return True
        if x is None:
            self.logger.debug("Lines are parallel – no conflict")
            return True

        self.logger.debug(
            "Lines intersect at (x=%s, z=%s) – potential conflict", x, z
        )
        return False

    def _calc_angle_relation(self, c1: Chain, c2: Chain, r_pole: Pole) -> None:
        """
        Determine and set the angle factor of ``c2`` relative to ``c1``.

        The computation depends on whether any absolute pole lies at infinity
        and on the displacement vectors of the chains to the relative pole.

        Parameters
        ----------
        c1, c2 : Chain
            Chain ``c1`` is the reference, ``c2`` receives the computed factor.
        r_pole : Pole
            The relative pole common to both chains.
        """
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)

        self.logger.debug(
            "Calculating angle relation: c%d = factor * c%d", c2_idx, c1_idx
        )

        c1_dist = c1.displacement_to_rpoles
        c2_dist = c2.displacement_to_rpoles

        # Cases where at least one displacement is undefined (infinite pole)
        if c1_dist is None or c2_dist is None:
            self.logger.info("At least one chain has an infinite absolute "
                             "pole")
            if c1_dist is None:
                self.logger.debug("Chain %d has infinite absolute pole",
                                  c1_idx)
                if c1.angle_factor in (0, 1):
                    c1.angle_factor = 1
                    self.logger.debug(
                        "Set angle_factor of chain %d to 1 (fallback)",
                        c1_idx
                    )
                a = np.array([[c1.absolute_pole.node.x],
                              [c1.absolute_pole.node.z]])
                l1 = r_pole.coords - a
                r21 = c2_dist[r_pole]
                factor = (
                                 np.linalg.norm(l1) / np.linalg.norm(r21)
                         ) * np.sign(np.dot(l1.T, r21)).item()
                c2.angle_factor = factor
                self.logger.info(
                    "Set angle_factor of chain %d to %s "
                    "(derived from geometry)",
                    c2_idx,
                    factor,
                )
            if c2_dist is None:
                self.logger.debug("Chain %d has infinite absolute pole",
                                  c2_idx
                                  )
                if c2.angle_factor == 0:
                    c2.angle_factor = 1
                    self.logger.debug(
                        "Set angle_factor of chain %d to 1 (fallback)",
                        c2_idx
                    )
            return

        # rPole itself is at infinity – treat as translation
        if r_pole.is_infinite:
            self.logger.info("Relative pole is infinite – "
                             "treating as translation")
            if c2.angle_factor == 0:
                c2.angle_factor = 1
                self.logger.debug(
                    "Set angle_factor of chain %d to 1 (infinite rPole)",
                    c2_idx
                )
            return

        # Normal case: both chains have finite displacements
        l1 = c1_dist[r_pole]
        r21 = c2_dist[r_pole]
        factor = (
                         np.linalg.norm(l1) / np.linalg.norm(r21)
                 ) * np.sign(np.dot(l1.T, r21)).item()
        c2.angle_factor = factor
        self.logger.info(
            "Computed angle_factor for chain %d: %s "
            "(ratio of displacement vectors)",
            c2_idx,
            factor,
        )

    @staticmethod
    def _get_rPole_from_chain(node: Node, chain: Chain) -> Optional[Pole]:
        """
        Retrieve the relative pole belonging to ``node`` from ``chain``.

        Parameters
        ----------
        node : Node
            The node for which the relative pole is searched.
        chain : Chain
            The chain that should contain the relative pole.

        Returns
        -------
        Pole or None
            The matching relative pole, or ``None`` if not found.
        """
        for r_pole in chain.relative_pole:
            if r_pole.node == node:
                return r_pole
        return None

    def __call__(self) -> bool:
        """
        Run Validation directly via callable instance syntax.
        """
        self.logger.debug("Validator invoked via __call__")
        return self.run()


@dataclass(eq=False)
class AngleCalculator(LoggerMixin):
    chains: List[Chain]
    node_to_chains: Dict[Node, List[Chain]]

    debug: bool = False

    def calculate_angles(self, target_chain: Chain, target_angle: float) -> \
            None:
        print('---------------------------')
        print('Schritt 4: Winkelberechnung')
        print('---------------------------')
        print(f'Die Drehwinkel aller Scheiben sollen so bestimmt werden, '
              f'dass die Scheibe {self.chains.index(target_chain)} den '
              f'Winkel von {target_angle} hat.')

        angle, target_chain_index = (self._init_target_chain(
            target_chain, target_angle
        ))

        print('Berechnung des Winkels von Scheibe 0, so dass,')
        print(f'Scheibe {target_chain_index} den Winkel {angle} hat.')

        if target_chain_index != 0:
            self._backward_calc(target_chain_index, angle)
        else:
            print(' -> Keine Rückwärtsberechnung nötig, da die ausgewählte '
                  'Scheibe Scheibe 0 ist!')
        self._forward_calc()

        print('-----------------------')
        print('Schritt 4 abgeschlossen')
        print('-----------------------')

    def _init_target_chain(self, target_chain: Chain, target_angle: float):
        # Behandlung der ersten Scheibe, falls sie nicht Ziel ist und nicht
        # starr
        target_chain_index = self.chains.index(target_chain)
        chain_0 = self.chains[0]
        if chain_0 != target_chain and not chain_0.stiff:
            chain_0.angle_factor = 1
            print('Set angle_factor=1 for chain 0')

        # Behandlung, falls target_chain starr ist
        if target_chain.stiff:
            print(f'Die Ausgewählte Scheibe mit dem Index:'
                  f' {target_chain_index} ist starr!')
            print(' -> Keinen Drehwinkel!')
            target_chain_index += 1
            print(f' -> neue Scheibe mit dem Index {target_chain_index}')
            target_chain = self.chains[target_chain_index]
            # TODO: Warum wir der Winkel hier auf -1 oder +1 gesetzt?
            if not target_chain.absolute_pole.is_infinite:
                target_angle = -1
            else:
                target_angle = 1

        target_chain.angle = target_angle

        return target_angle, target_chain_index

    def _backward_calc(self, target_chain_index: int, initial_angle: float) \
            -> None:
        # TODO: das geht nur, wenn die Reihenfolge der Liste mit der der
        #  Geometrie übereinstimmt ! Das muss vorher sichergestellt sein!
        angle = initial_angle
        for i in range(target_chain_index - 1, -1, -1):
            next_chain = self.chains[i + 1]
            factor = next_chain.angle_factor
            print(f'Berechnung für Scheibe {i}: factor = {factor}')
            if factor == 0:
                print(f"Winkelberechnung für Kette {i + 1} abgebrochen: "
                      f"angle_factor = 0")
                break

            current_chain = self.chains[i]
            if current_chain.angle_factor == 0:
                print(f"Winkelberechnung für Kette {i} abgebrochen: "
                      f"angle_factor = 0")
                break

            angle /= factor
            current_chain.angle = angle
            print(f"Berechneter Winkel für Scheibe {i}: {angle}")

    def _forward_calc(self) -> None:
        print('(((((((((((((())))))))))))))')
        print('Berechnung aller Scheibenwinkel')

        previous_chain = None
        for node, chains in self.node_to_chains.items():
            print('=========================')
            print(f'Knoten: {node.x}, {node.z}')
            print(f'Überprüfte Scheiben: '
                  f'{[self.chains.index(c) for c in chains]}')

            pairs = self._get_chain_pairs(chains, previous_chain)

            for c1, c2 in pairs:
                self._calc_c2_angle_from_c1(c1, c2)
                previous_chain = c2

    @staticmethod
    def _get_chain_pairs(chains: List[Chain], previous_chain: Chain) -> (
            List)[Tuple[Chain, Chain]]:
        """Gets pairs of chains for angle calculation."""
        if previous_chain and previous_chain in chains:
            return [(previous_chain, c) for c in chains if c != previous_chain]
        else:
            return list(combinations(chains, 2))

    def _calc_c2_angle_from_c1(self, c1: Chain, c2: Chain) -> bool:
        c1_idx = self.chains.index(c1)
        c2_idx = self.chains.index(c2)
        print(f' -> Kombo: {c1_idx} - {c2_idx}')
        if c1.stiff or c2.stiff:
            print('stiff')
            return False

        angle = c1.angle * c2.angle_factor
        c2.angle = angle

        print(f'   -> c{c2_idx}.angle = c{c1_idx}.angle *  '
              f'c{c2_idx}.angle_factor')
        print(f'   -> c{c2_idx}.angle = {c1.angle} * {c2.angle_factor}')
        print(f'   -> c{c2_idx}.angle = {angle}')
        return True


@dataclass(eq=False)
class DisplacementCalculator(LoggerMixin):
    chains: List[Chain]
    bars: List[Bar]
    node_to_chains: Dict[Node, List[Chain]]

    debug: bool = False

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
            for conn_chain in self.node_to_chains[rPole.node]:
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
