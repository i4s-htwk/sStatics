
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.loads import BarPointLoad, NodePointLoad
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.temperature import BarTemp


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
    mesh : tuple[Bar, ...]
        The bars of the system after segmentation.
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

    def node_to_bar_map(self, segmented: bool = True):
        bars = self.mesh if segmented else self.bars
        node_connection = {}
        for bar in bars:
            for node in (bar.node_i, bar.node_j):
                if node not in node_connection:
                    node_connection[node] = []
                node_connection[node].append(bar)
        return node_connection

    def nodes(self, segmented: bool = True):
        return list(self.connected_nodes(segmented=segmented).keys())

    @property
    def mesh(self):
        return self._mesh.mesh

    @property
    def user_mesh(self):
        return self._mesh.user

    def mesh_segments_of(self, bar):
        return self._mesh.mesh_segments_of(bar)

    def user_segments_of(self, bar: Bar) -> List[Bar]:
        return self._mesh.user_segments_of(bar)

    def bar_of(self, mesh_segment: Bar) -> Bar:
        return self._mesh.bar_of(mesh_segment)

    @property
    def polplan(self):
        if not hasattr(self, "_polplan"):
            from sstatics.core.preprocessing.poleplan import Polplan
            self._polplan = Polplan(self)
        return self._polplan

    def create_mesh(self, user_divisions=None):
        self._mesh = Mesh(bars=self.bars, user_divisions=user_divisions)()


@dataclass(eq=False)
class Mesh:
    bars: List[Bar]
    user_divisions: Optional[Dict[Bar, List[float]]] = None

    def __post_init__(self):
        self.bars = list(self.bars)
        self._mesh: List[Bar] = []
        self._user: List[Bar] = []

        self._map_bar_user: Dict[Bar, List[Bar]] = {}
        self._map_bar_mesh: Dict[Bar, List[Bar]] = {}

    @property
    def mesh(self):
        return self._mesh

    @property
    def user(self):
        return self._user

    @property
    def _map_mesh_bar(self):
        reverse_map = {}
        for bar, mesh in self._map_bar_mesh.items():
            for segment in mesh:
                reverse_map[segment] = bar
        return reverse_map

    def generate(self):
        user_divisions = self.user_divisions or {}
        calc_mesh = []
        user_mesh = []

        map_bar_user = {}
        map_bar_mesh = {}

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
            map_bar_user[bar] = user_segments

            calc_segments = user_segments.copy()

            if load_pos:
                for idx, pos_load in (
                        self._assign_loads_to_segments(
                            load_pos, user_pos).items()
                ):
                    if not set(pos_load.keys()).issubset({0, 1}):
                        calc_segments[idx:idx + 1] = (
                            self._split(calc_segments[idx], pos_load)
                        )
            map_bar_mesh[bar] = calc_segments
            calc_mesh.extend(calc_segments)

        self._user = user_mesh
        self._mesh = calc_mesh

        self._map_bar_user = map_bar_user
        self._map_bar_mesh = map_bar_mesh

    def mesh_segments_of(self, bar: Bar) -> List[Bar]:
        return self._map_bar_mesh.get(bar)

    def user_segments_of(self, bar: Bar) -> List[Bar]:
        return self._map_bar_user.get(bar)

    def bar_of(self, mesh_segment: Bar) -> Bar:
        return self._map_mesh_bar.get(mesh_segment)

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
class SystemModifier:
    """
    Modifies a given structural system by tracking and applying changes such as
    support modifications, bar hinge insertions, load deletions, and more.

    This class maintains a mapping between original and modified components
    (bars and nodes), making it possible to incrementally update the system
    while preserving references to the original configuration. It is especially
    useful in structural mechanics for performing operations like virtual
    force methods or stepwise modifications for influence line generation.

    Parameters
    ----------
    system : :any:`System`
        The structural system to be modified. This must include bars and nodes
        as fundamental elements, potentially with loads, supports, and other
        mechanical properties.

    Attributes
    ----------
    bar_map : dict[:any:`Bar`, Optional[:any:`Bar`]]
        A mapping from the original bars to their current (potentially
        modified) versions.
        This ensures that modifications can always refer back to the
        original structure. If a bar has been deleted, it is mapped to `None`.

        This is essential for managing modifications over time without losing
        the relationship between system states.

    node_map : dict[:any:`Node`, :any:`Node`]
        Maps original nodes to the current (possibly modified) versions.
        Used to consistently track changes in node-based properties like
        displacements, loads, and support conditions. The original node always
        remains the key.

        This map ensures that new system configurations reflect changes such as
        support removals or load applications, while maintaining link to the
        original nodes.

    memory_modification : list[tuple[Union[:any:`Bar`, :any:`Node`], str]]
        Stores a log of all applied structural modifications. Each entry
        consists of an object (either a `Bar` or `Node`) and a string that
        describes what degree of freedom was modified (e.g., 'u', 'w', 'phi',
        or 'hinge_phi_i').

        This is especially useful when constructing related systems later, for
        example when using the principle of virtual forces to create unit load
        systems corresponding to removed constraints.

    memory_bar_point_load : dict[:any:`Bar`, list[float]]
        Records bars that have point loads acting at positions strictly between
        0 and 1 (non-endpoint). The list of floats denotes the relative
        position of each such load along the bar.

        This is used to ensure that when systems are later subdivided or
        aligned (e.g., for assembling superposition systems or plotting
        influence lines), bars can be split consistently at these key
        locations.

        The positions are automatically collected in `__post_init__` and
        updated when new point loads are added via modification methods.
    """

    system: System

    def __post_init__(self):
        self.bar_map = {bar: bar for bar in self.system.bars}
        self.node_map = {node: node for node in self.system.nodes()}
        self.memory_modification: list[tuple[Union[Bar, Node], str]] = []
        self.memory_bar_point_load: dict[Bar, list[float]] = {}
        self._initialize_bar_point_positions()

    def _initialize_bar_point_positions(self):
        """Initializes memory of internal point loads on bars.

        Scans through all bars in the system and stores the position of any
        internal point loads that are located strictly between the two bar ends
        (i.e., 0 < position < 1). These stored positions are later used for
        consistent bar splitting when systems are modified in parallel.

        Called during post-initialization to ensure this data is available
        immediately after instantiation.
        """
        for bar in self.system.bars:
            loads = bar.point_loads
            positions = [pl.position for pl in loads if 0 < pl.position < 1]
            if positions:
                self.memory_bar_point_load[bar] = positions

    def modify_bar_force_influ(self, obj: Bar,
                               force: Literal['fx', 'fz', 'fm'],
                               position: float, virt_force: float = 1
                               ) -> System:
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

    def modify_bar_deform_influ(
            self, obj: Bar, deform: Literal['u', 'w', 'phi'],
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

    def modify_node_force_influ(
            self, obj: Node, force: Literal['fx', 'fz', 'fm'],
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

    def _update_bar(self, original_bar: Bar, new_bar: Bar):
        """Updates the internal bar mapping after a modification.

        Replaces the mapped value in `bar_map` associated with a given
        original bar by the newly modified version. If the original bar was
        not previously mapped, it is added.

        Parameters
        ----------
        original_bar : :any:`Bar`
            The original unmodified bar used as a reference key.

        new_bar : :any:`Bar`
            The new version of the bar with modifications applied.
        """
        for orig, current in self.bar_map.items():
            if current == original_bar:
                self.bar_map[orig] = new_bar
                return
        self.bar_map[original_bar] = new_bar

    def _get_current_bar(self, original_bar: Bar):
        """Retrieves the current version of a bar after potential
        modifications.

        If the original bar was modified or deleted, returns the updated bar or
        raises an error if the bar no longer exists.

        Parameters
        ----------
        original_bar : :any:`Bar`
            The original bar as a reference.

        Returns
        -------
        :any:`Bar`
            The currently valid version of the bar in the system.

        Raises
        ------
        ValueError
            If the bar has been deleted from the system.
        """
        current = self.bar_map.get(original_bar, original_bar)
        if current is None:
            raise ValueError("This bar has been deleted from the system.")
        return current

    def _get_current_node(self, original_node):
        """Retrieves the current version of a node after modifications.

        Looks up the original node in the internal node map and returns the
        corresponding updated node if one exists.

        Parameters
        ----------
        original_node : :any:`Node`
            The original node used as a key in the map.

        Returns
        -------
        :any:`Node`
            The modified node if present, otherwise the original.
        """
        return self.node_map.get(original_node, original_node)

    def _find_bar_index(self, bars: list[Bar], target: Bar):
        """Finds the index of a target bar in a list based on its nodal
        coordinates.

        The method matches bars by comparing the coordinates of their start and
        end nodes. Useful when a bar object has changed identity due to
        mutation or reconstruction.

        Parameters
        ----------
        bars : list[:any:`Bar`]
            List of bars to search within.

        target : :any:`Bar`
            The bar whose index is being searched.

        Returns
        -------
        int
            Index of the target bar.

        Raises
        ------
        ValueError
            If the bar cannot be found in the list.
        """
        for i, bar in enumerate(bars):
            if (
                    bar.node_i.x == target.node_i.x and
                    bar.node_i.z == target.node_i.z and
                    bar.node_j.x == target.node_j.x and
                    bar.node_j.z == target.node_j.z
            ):
                return i
        raise ValueError("Matching bar not found in system.")

    def delete_loads(self):
        """Removes all loads from the current system.

        This includes both nodal and bar loads (point and line loads) as well
        as thermal effects. It preserves the topology and geometry of the
        system, but clears external influences.

        The system is reconstructed with the cleared nodes and bars. Also
        updates the internal mappings accordingly.

        Returns
        -------
        :any:`System`
            The updated system with all loads removed.
        """
        bars = list(self.system.bars)
        cleared_nodes = {}

        for i, bar in enumerate(bars):
            if bar is None:
                continue

            orig_bar = next((orig for orig, mod in self.bar_map.items()
                             if mod == bar), bar)

            node_i = self.node_map.get(bar.node_i, bar.node_i)
            if bar.node_i in cleared_nodes:
                cleared_node_i = cleared_nodes[bar.node_i]
            else:
                cleared_node_i = replace(node_i, displacements=(), loads=())
                cleared_nodes[bar.node_i] = cleared_node_i
                self.node_map[bar.node_i] = cleared_node_i

            node_j = self.node_map.get(bar.node_j, bar.node_j)
            if bar.node_j in cleared_nodes:
                cleared_node_j = cleared_nodes[bar.node_j]
            else:
                cleared_node_j = replace(node_j, displacements=(), loads=())
                cleared_nodes[bar.node_j] = cleared_node_j
                self.node_map[bar.node_j] = cleared_node_j

            cleared_bar = replace(
                bar,
                node_i=cleared_node_i,
                node_j=cleared_node_j,
                line_loads=(),
                point_loads=(),
                temp=BarTemp(temp_o=0, temp_u=0),
            )

            self.bar_map[orig_bar] = cleared_bar
            bars[i] = cleared_bar

        self.system = System(bars)
        return self.system

    def insert_hinge(
            self,
            bar_obj: Bar,
            hinge: Literal[
                'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
                'hinge_u_j', 'hinge_w_j', 'hinge_phi_j'
            ]
    ):
        """Inserts a hinge at a specific end and direction of a bar.

        Modifies the selected bar by enabling a hinge for axial (u), shear (w),
        or moment (phi) behavior at the start (`_i`) or end (`_j`). Updates
        the system and internal memory for reconstruction or analysis.

        Parameters
        ----------
        bar_obj : :any:`Bar`
            The target bar to modify.

        hinge : str
            The type and location of hinge, chosen from:
            'hinge_u_i', 'hinge_w_i', 'hinge_phi_i',
            'hinge_u_j', 'hinge_w_j', 'hinge_phi_j'.

        Returns
        -------
        :any:`System`
            The updated system with the hinge applied.

        Raises
        ------
        ValueError
            If the hinge is already present on the bar.
        """
        current_bar = self._get_current_bar(bar_obj)

        if getattr(current_bar, hinge):
            raise ValueError(f"{hinge} is already present on the bar.")

        node_i = self._get_current_node(current_bar.node_i)
        node_j = self._get_current_node(current_bar.node_j)

        modified_bar = replace(
            current_bar,
            node_i=node_i,
            node_j=node_j,
            **{hinge: True}
        )

        bars = list(self.system.bars)
        idx = self._find_bar_index(bars, current_bar)
        bars[idx] = modified_bar
        self.system = System(bars)

        self._update_bar(bar_obj, modified_bar)
        self.memory_modification.append((bar_obj, hinge))

        return self.system

    def modify_support(
            self,
            node_obj: Node,
            support: Literal['u', 'w', 'phi']
    ):
        """Frees a specific degree of freedom (support) at a given node.

        Modifies the boundary condition of the node, turning the specified
        support direction into a free (unconstrained) one. The system is
        updated accordingly, and all bars connected to the node are
        reconstructed.

        Parameters
        ----------
        node_obj : :any:`Node`
            The node whose support is being modified.

        support : {'u', 'w', 'phi'}
            Direction of the support to release:
            'u' for horizontal,
            'w' for vertical,
            'phi' for rotational.

        Returns
        -------
        :any:`System`
            The system with the updated boundary condition.

        Raises
        ------
        ValueError
            If the selected support is already free.
        """
        node = self._get_current_node(node_obj)

        if getattr(node, support) == 'free':
            raise ValueError(f"Support '{support}' is already free.")

        modified_node = replace(node, **{support: 'free'})
        bars = list(self.system.bars)

        for bar in bars:
            updated_bar = None

            if bar.node_i.x == node.x and bar.node_i.z == node.z:
                updated_bar = replace(bar, node_i=modified_node)
            elif bar.node_j.x == node.x and bar.node_j.z == node.z:
                updated_bar = replace(bar, node_j=modified_node)

            if updated_bar:
                bars[bars.index(bar)] = updated_bar
                self._update_bar(bar, updated_bar)

        self.system = System(bars)
        self.node_map[node_obj] = modified_node
        self.memory_modification.append((node_obj, support))

        return self.system

    def modify_node_force_vir(
            self, obj: Node, force: Literal['fx', 'fz', 'fm'],
            virt_force: float = 1
    ):
        """Applies a virtual load to a node.

        This is used for virtual work or energy methods (e.g., to compute
        displacements). It modifies the node by assigning a virtual point load
        in the specified direction.

        Parameters
        ----------
        obj : :any:`Node`
            The node to which the virtual load is applied.

        force : {'fx', 'fz', 'fm'}
            Type of force:
            'fx' = horizontal,
            'fz' = vertical,
            'fm' = moment.

        virt_force : float, optional
            Magnitude of the virtual force (default is 1.0).

        Returns
        -------
        :any:`System`
            Updated system with the virtual load applied to the node.
        """
        current_node = self.node_map.get(obj)
        if current_node is None:
            raise ValueError("Node not found in system")

        load = NodePointLoad(
            x=virt_force if force == 'fx' else 0.0,
            z=virt_force if force == 'fz' else 0.0,
            phi=virt_force if force == 'fm' else 0.0
        )

        modified_node = replace(current_node, loads=load)
        bars = list(self.system.bars)
        new_bars = []

        for bar in bars:
            if bar.node_i == current_node or bar.node_j == current_node:
                new_bars.append(replace(
                    bar,
                    node_i=modified_node if bar.node_i == current_node
                    else bar.node_i,
                    node_j=modified_node if bar.node_j == current_node
                    else bar.node_j
                ))
            else:
                new_bars.append(bar)

        self.node_map[obj] = modified_node
        self.system = System(new_bars)
        return self.system

    def modify_bar_force_vir(
            self, obj: Bar, force: Literal['fx', 'fz', 'fm'],
            virt_force: float = 1, position: float = 0):
        """Applies a virtual point load to a bar.

        Useful for virtual force methods to calculate internal forces
        (e.g., to determine internal actions). The load is placed at the
        specified relative position along the bar.

        Parameters
        ----------
        obj : :any:`Bar`
            The bar where the virtual load is applied.

        force : {'fx', 'fz', 'fm'}
            Direction/type of load:
            'fx' = axial,
            'fz' = transverse,
            'fm' = moment.

        position : float, optional
            Relative position along the bar (0 to 1), default is 0.

        virt_force : float, optional
            Magnitude of the virtual force, default is 1.

        Returns
        -------
        :any:`System`
            Updated system with the virtual bar load.
        """
        current_bar = self.bar_map.get(obj)
        if current_bar is None:
            raise ValueError("Bar not found in system")

        load = BarPointLoad(
            x=virt_force if force == 'fx' else 0.0,
            z=virt_force if force == 'fz' else 0.0,
            phi=virt_force if force == 'fm' else 0.0,
            position=position
        )

        modified_bar = replace(current_bar, point_loads=load)
        bars = list(self.system.bars)
        for i, bar in enumerate(bars):
            if bar == current_bar:
                bars[i] = modified_bar
                break

        self.bar_map[obj] = modified_bar
        if 0 < position < 1:
            self.memory_bar_point_load.setdefault(obj, []).append(position)

        self.system = System(bars)
        return self.system

    def delete_bar(self, bar_obj: Bar):
        """Removes a bar from the system.

        Updates the internal `bar_map` to mark the bar as deleted and
        reconstructs the system without the specified bar.

        Parameters
        ----------
        bar_obj : :any:`Bar`
            The bar to be removed.

        Returns
        -------
        :any:`System`
            The system without the specified bar.

        Raises
        ------
        ValueError
            If the bar has already been deleted or cannot be found.
        """
        if self.bar_map.get(bar_obj) is None:
            raise ValueError("Bar has already been deleted.")

        bars = list(self.system.bars)
        try:
            bars.remove(self._get_current_bar(bar_obj))
        except ValueError:
            raise ValueError("The specified bar was not found in the "
                             "current system.")

        self._update_bar(bar_obj, None)
        self.system = System(bars)
        return self.system

    def create_node_load_systems(self):
        """Generates a set of systems with unit loads applied at modified DOFs.

        For each recorded modification (either release of support or insertion
        of hinge), a system is constructed where a unit point load is applied
        at the corresponding degree of freedom (x, z, or phi). This is useful
        for the :py:class:`KGV`.

        Returns
        -------
        list[:any:`System`]
            A list of systems, each with one unit node load applied.
        """
        systems = []

        for obj, name in self.memory_modification:
            if isinstance(obj, Bar):
                node = obj.node_i if '_i' in name else obj.node_j
                dof = 'x' if 'u' in name else 'z' if 'w' in name else 'phi'
            elif isinstance(obj, Node):
                node = obj
                dof = 'x' if name == 'u' else 'z' if name == 'w' else 'phi'
            else:
                continue

            current_node = self._get_current_node(node)

            load = NodePointLoad(
                x=1.0 if dof == 'x' else 0.0,
                z=1.0 if dof == 'z' else 0.0,
                phi=1.0 if dof == 'phi' else 0.0
            )

            new_node = replace(current_node, loads=load)

            new_bars = [replace(b,
                                node_i=new_node if b.node_i == current_node
                                else b.node_i,
                                node_j=new_node if b.node_j == current_node
                                else b.node_j)
                        for b in self.system.bars]

            systems.append(System(new_bars))

        return systems

    def division_positions_mesh(self):
        """
        Returns a dictionary mapping each bar to a
        list of relative positions (0 < pos < 1) where point loads exist.
        This is compatible with the `user_divisions` argument of
        `MeshGenerator`.

        Returns
        -------
        dict[Bar, list[float]]
            A dictionary where keys are Bar objects and values are lists of
            division positions on that bar.
        """
        result = {}
        for original_bar, positions in self.memory_bar_point_load.items():
            bar = self._get_current_bar(original_bar)
            if bar not in result:
                result[bar] = []
            result[bar].extend(positions)
        return result
