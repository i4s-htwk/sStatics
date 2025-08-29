
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.loads import NodePointLoad
from sstatics.core.preprocessing.node import Node


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
        nodes = self.nodes(mesh_type='bars')
        for i, node in enumerate(nodes[0:-1]):
            for other_node in nodes[i + 1:]:
                if node.same_location(other_node) and node != other_node:
                    raise ValueError(
                        'Inconsistent system. Nodes with the same location '
                        'need to be the same instance.'
                    )
        to_visit, visited = [nodes[0]], []
        while to_visit:
            curr_node = to_visit.pop(0)
            if curr_node not in visited:
                visited.append(curr_node)
                to_visit += self.connected_nodes(mesh_type='bars')[curr_node]
        if set(visited) != set(nodes):
            raise ValueError("The system's graph needs to be connected.")

        self.create_mesh(user_divisions=self.user_divisions)

    def connected_nodes(
            self, mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh'
    ):
        bar_type = self.__getattribute__(mesh_type)
        connections = {}
        for bar in bar_type:
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

    def nodes(self, mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh'):
        return list(self.connected_nodes(mesh_type=mesh_type).keys())

    @property
    def max_dimensions(self) -> Tuple[float, float]:
        nodes = self.nodes(mesh_type='bars')
        x_coords = [node.x for node in nodes]
        z_coords = [node.z for node in nodes]
        return max(x_coords) - min(x_coords), max(z_coords) - min(z_coords)

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
            from sstatics.core.preprocessing.poleplan.objects import Poleplan
            self._polplan = Poleplan(self)
        return self._polplan

    def create_mesh(self, user_divisions=None):
        self._mesh = Mesh(bars=self.bars, user_divisions=user_divisions)()


@dataclass(eq=False)
class Mesh:
    r"""Represents a mesh of bars.

    Attributes
    ----------
    bars : List[Bar]
        A list of bars to be meshed.
    user_divisions : Optional[Dict[Bar, List[float]]]
        Optional user-defined divisions for each bar.

    Notes
    -----
    The `user_divisions` dictionary maps each bar to a list of positions
    where the bar should be divided.
    """
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
        r"""Gets the generated mesh.

        Returns
        -------
        List[Bar]
            The list of bars in the mesh.
        """
        return self._mesh

    @property
    def user(self):
        r"""Gets the user-defined mesh.

        Returns
        -------
        List[Bar]
            The list of bars in the user-defined mesh.
        """
        return self._user

    @property
    def _map_mesh_bar(self):
        r"""Gets a mapping from mesh bars to their original bars.

        Returns
        -------
        Dict[Bar, Bar]
            A dictionary mapping each mesh bar to its original bar.
        """
        reverse_map = {}
        for bar, mesh in self._map_bar_mesh.items():
            for segment in mesh:
                reverse_map[segment] = bar
        return reverse_map

    def generate(self):
        r"""Generates the mesh based on the input bars and user divisions.

        This method populates the `mesh` and `user` properties.

        Returns
        -------
        None

        Notes
        -----
        The mesh generation process involves splitting the input bars into
        segments based on user-defined divisions and point loads.
        """
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
        r"""Gets the mesh segments corresponding to a given bar.

        Parameters
        ----------
        bar : Bar
            The bar for which to retrieve mesh segments.

        Returns
        -------
        List[Bar]
            The list of mesh segments corresponding to the given bar.
        """
        return self._map_bar_mesh.get(bar)

    def user_segments_of(self, bar: Bar) -> List[Bar]:
        r"""Gets the user-defined segments corresponding to a given bar.

        Parameters
        ----------
        bar : Bar
            The bar for which to retrieve user-defined segments.

        Returns
        -------
        List[Bar]
            The list of user-defined segments corresponding to the given bar.
        """
        return self._map_bar_user.get(bar)

    def bar_of(self, mesh_segment: Bar) -> Bar:
        r"""Gets the original bar corresponding to a given mesh segment.

        Parameters
        ----------
        mesh_segment : Bar
            The mesh segment for which to retrieve the original bar.

        Returns
        -------
        Bar
            The original bar corresponding to the given mesh segment.
        """
        return self._map_mesh_bar.get(mesh_segment)

    @staticmethod
    def _assign_loads_to_segments(load_pos_i, user_pos_i):
        r"""Assigns loads to segments based on their positions.

        Parameters
        ----------
        load_pos_i : Dict[float, List[NodePointLoad]]
            A dictionary of loads at specific positions.
        user_pos_i : List[float]
            A list of user-defined positions.

        Returns
        -------
        Dict[int, Dict[float, List[NodePointLoad]]]
            A dictionary mapping segment indices to loads at specific
            positions.

        Notes
        -----
        The loads are assigned to segments based on their positions relative to
        the user-defined positions.
        """
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
        r"""Gets the point loads on a given bar.

        Parameters
        ----------
        bar : Bar
            The bar for which to retrieve point loads.

        Returns
        -------
        Dict[float, List[NodePointLoad]]
            A dictionary mapping positions to lists of point loads.
        """
        return {load.position: [load] for load in bar.point_loads}

    @staticmethod
    def _transfer_loads_to_user_pos(user_pos_i: list[float], load_pos_i):
        r"""Transfers loads to user-defined positions.

        Parameters
        ----------
        user_pos_i : List[float]
            A list of user-defined positions.
        load_pos_i : Dict[float, List[NodePointLoad]]
            A dictionary of loads at specific positions.

        Returns
        -------
        Tuple[Dict[float, List[NodePointLoad]], Dict[float,
            List[NodePointLoad]]]
            A tuple containing the updated user positions and remaining loads.
        """
        user_positions = defaultdict(list, {k: [] for k in user_pos_i})
        for pos in list(load_pos_i):
            if pos in user_positions or pos in (0.0, 1.0):
                user_positions[pos].extend(load_pos_i[pos])
                del load_pos_i[pos]

        return user_positions, load_pos_i

    @staticmethod
    def _add_end_loads(user_positions, bar):
        r"""Adds end loads to user-defined positions.

        Parameters
        ----------
        user_positions : Dict[float, List[NodePointLoad]]
            A dictionary of user-defined positions and user loads.
        bar : Bar
            The bar for which to add end loads.

        Returns
        -------
        Dict[float, List[NodePointLoad]]
            The updated dictionary of user-defined positions and loads.
        """
        for load in bar.point_loads:
            if load.position in (0.0, 1.0):
                user_positions[load.position].append(load)
        return user_positions

    @staticmethod
    def _to_node_load(point_loads):
        r"""Converts point loads to node loads.

        Parameters
        ----------
        point_loads : List[NodePointLoad]
            A list of point loads.

        Returns
        -------
        List[NodePointLoad]
            The list of node loads.
        """
        return [
            NodePointLoad(load.x, load.z, load.phi, load.rotation)
            for load in point_loads
        ]

    @staticmethod
    def _interp_coords(bar, position: float):
        r"""Interpolates coordinates at a given position on a bar.

        Parameters
        ----------
        bar : Bar
            The bar for which to interpolate coordinates.
        position : float
            The position on the bar.

        Returns
        -------
        Tuple[float, float]
            The interpolated x and z coordinates.
        """
        c, s = np.cos(bar.inclination), np.sin(bar.inclination)
        return (
            bar.node_i.x + c * position * bar.length,
            bar.node_i.z - s * position * bar.length
        )

    @staticmethod
    def _interp_lloads(bar: Bar,
                       prev_bar: Optional[Bar],
                       pos: Optional[float] = None):
        r"""Interpolates line loads on a bar.

        Parameters
        ----------
        bar : Bar
            The bar for which to interpolate line loads.
        prev_bar : Optional[Bar]
            The previous bar (if any).
        pos : Optional[float]
            The position on the bar (if any).

        Returns
        -------
        List[LineLoad]
            The list of interpolated line loads.
        """
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
        r"""Splits a bar into segments based on user-defined positions and
        loads.

        Parameters
        ----------
        bar : Bar
            The bar to be split.
        pos_dict : Dict[float, List[NodePointLoad]]
            A dictionary of positions and loads.

        Returns
        -------
        List[Bar]
            The list of split bars.
        """
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
        r"""Creates a new node at a given position on a bar.

        Parameters
        ----------
        bar : Bar
            The bar on which to create the node.
        position : float
            The position on the bar.
        point_loads : List[NodePointLoad]
            The point loads at the node.

        Returns
        -------
        Node
            The created node.
        """
        x, z = self._interp_coords(bar, position)
        return Node(x, z, loads=self._to_node_load(point_loads))

    def _bar_first(self, bar, node_j, pos, end_point_loads):
        r"""Creates the first segment of a bar.

        Parameters
        ----------
        bar : Bar
            The original bar.
        node_j : Node
            The end node of the segment.
        pos : float
            The position of the end node.
        end_point_loads : List[BarPointLoad]
            The point loads at the end node.

        Returns
        -------
        Bar
            The created bar segment.
        """
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
        r"""Creates a middle segment of a bar.

        Parameters
        ----------
        bar : Bar
            The original bar.
        prev_bar : Bar
            The previous bar segment.
        node_j : Node
            The end node of the segment.
        pos : float
            The position of the end node.

        Returns
        -------
        Bar
            The created bar segment.
        """
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
        r"""Creates the last segment of a bar.

        Parameters
        ----------
        bar : Bar
            The original bar.
        prev_bar : Bar
            The previous bar segment.
        end_point_loads : List[BarPointLoad]
            The point loads at the end node.

        Returns
        -------
        Bar
            The created bar segment.
        """
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
        r"""Generates the mesh and returns the mesh object.

        Returns
        -------
        Mesh
            The mesh object.
        """
        self.generate()
        return self
