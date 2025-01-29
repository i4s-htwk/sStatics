
from dataclasses import dataclass
from functools import cache

from sstatics.core import Bar


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
    def nodes(self, segmented: bool = True):
        return list(self.connected_nodes(segmented=segmented).keys())
