
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional

import numpy as np

from sstatics.core.preprocessing.bar import Bar, BarSecond
from sstatics.core.preprocessing.loads import NodePointLoad
from sstatics.core.preprocessing.node import Node


@dataclass(eq=False)
class System:
    """Represents a statical system composed of interconnected bars.

    This class models a mechanical system made up of interconnected bars, where
    each bar connects two nodes.

    Parameters
    ----------
    bars : tuple[ :any:`Bar`, ...] | list[ :any:`Bar`] \
            | tuple[ :any:`BarSecond`, ...] | list[:any:`BarSecond`]
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
    mesh : tuple[:any:`Bar`, ...] | tuple[:any:`BarSecond`, ...]
        The bars of the system after segmentation.
    """

    bars: tuple[Bar, ...] | list[Bar] | tuple[BarSecond, ...] | list[BarSecond]
    user_divisions: (Optional[Dict[Bar, List[float]]]
                     | Optional[Dict[BarSecond, List[float]]]) = None

    def __post_init__(self):
        self.bars = tuple(self.bars)
        self._validate_geometry()
        self._validate_connectivity()
        self._validate_crossings()
        self.create_mesh(user_divisions=self.user_divisions)

    def _validate_geometry(self) -> None:
        """Verify the basic geometric consistency of the system.

        Checks
        ------
        * **Non-empty bar list** – raises if ``self.bars`` contains no
          element.
        * **Duplicate bars** – raises when two distinct ``Bar`` objects
          occupy the identical geometric location (both end nodes coincide).
          The check uses ``Bar.same_location`` which must compare the
          coordinates of the two end nodes.
        * **Node aliasing** – raises if two *different* ``Node`` instances
          share the same spatial coordinates while not being the same
          object. This prevents the situation where the same point in space
          is represented by multiple node objects, which would break the
          connectivity logic.

        Raises
        ------
        ValueError
            If any of the above conditions are violated.
        """
        if len(self.bars) == 0:
            raise ValueError("There need to be at least one bar.")
        for i, bar in enumerate(self.bars[0:-1]):
            if any(
                    bar.same_location(other_bar)
                    for other_bar in self.bars[i + 1:]
            ):
                raise ValueError(
                    "Cannot instantiate a system with bars that share the "
                    "same location."
                )
        nodes = self.nodes(mesh_type="bars")
        for i, node in enumerate(nodes[0:-1]):
            for other_node in nodes[i + 1:]:
                if node.same_location(other_node) and node != other_node:
                    raise ValueError(
                        "Inconsistent system. Nodes with the same location "
                        "need to be the same instance."
                    )

    def _validate_connectivity(self) -> None:
        """Ensure that the bar network forms a single connected component.

        The method performs a breadth-first search starting from the first
        node returned by ``self.nodes(mesh_type='bars')``. All nodes
        reachable via ``self.connected_nodes`` are collected in ``visited``.
        After the search finishes, the set of visited nodes must equal the
        set of all nodes; otherwise, the graph contains at least one
        isolated component.

        Raises
        ------
        ValueError
            If the system graph is not fully connected.
        """
        nodes = self.nodes(mesh_type="bars")
        to_visit, visited = [nodes[0]], []
        while to_visit:
            curr_node = to_visit.pop(0)
            if curr_node not in visited:
                visited.append(curr_node)
                to_visit += self.connected_nodes(mesh_type="bars")[curr_node]
        if set(visited) != set(nodes):
            raise ValueError("The system's graph needs to be connected.")

    def _validate_crossings(self) -> None:
        r"""Detect interior intersections between any pair of bars.

        Two bars are allowed to meet at a common node (i.e. share an
        endpoint). Any other intersection – including collinear overlap
        that is not a shared endpoint – raises an error.

        Implementation details
        ----------------------
        * **Axis-aligned bounding boxes (AABB)** are computed for each bar.
          Pairs whose boxes do not overlap are discarded before the more
          expensive orientation test.
        * **Orientation test** (signed triangle area) determines on which
          side of a directed segment a point lies. The classic “general
          case” condition

          .. math::

              (o_1 \cdot o_2 < 0) \;\wedge\; (o_3 \cdot o_4 < 0)

          holds exactly when the interiors of the two segments intersect.
        * **Collinear special cases** are handled by ``on_segment`` to catch
          overlapping but non-identical bars.

        Raises
        ------
        ValueError
            If any two bars intersect in their interiors. The error message
            contains the two offending ``Bar`` objects.
        """

        def _aabb(bar):
            xs = (bar.node_i.x, bar.node_j.x)
            zs = (bar.node_i.z, bar.node_j.z)
            return min(xs), max(xs), min(zs), max(zs)

        def orient(a: Node, b: Node, r: Node) -> float:
            """
            Signed area of triangle (a,b,r). Positive → counter-clockwise.
            """
            return (b.x - a.x) * (r.z - a.z) - (b.z - a.z) * (r.x - a.x)

        def on_segment(a: Node, b: Node, r: Node) -> bool:
            """Return True if b lies on the closed segment pr (collinear)."""
            return (
                    min(a.x, r.x) <= b.x <= max(a.x, r.x)
                    and min(a.z, r.z) <= b.z <= max(a.z, r.z)
            )

        def segments_intersect(b1: Bar, b2: Bar) -> bool:
            i1, j1 = b1.node_i, b1.node_j
            i2, j2 = b2.node_i, b2.node_j

            # Ignore intersection at a shared node
            if i1 in (i2, j2) or j1 in (i2, j2):
                return False

            o1 = orient(i1, j1, i2)
            o2 = orient(i1, j1, j2)
            o3 = orient(i2, j2, i1)
            o4 = orient(i2, j2, j1)

            # General case
            if (o1 * o2 < 0) and (o3 * o4 < 0):
                return True

            # Collinear special cases
            if o1 == 0 and on_segment(i1, i2, j1):
                return True
            if o2 == 0 and on_segment(i1, j2, j1):
                return True
            if o3 == 0 and on_segment(i2, i1, j2):
                return True
            if o4 == 0 and on_segment(i2, j1, j2):
                return True

            return False

        # First perform a cheap AABB (axis-aligned bounding box) check to
        # quickly reject bar pairs that cannot possibly intersect. Only if
        # their bounding boxes overlap do we proceed to the more expensive
        # segment intersection test.
        boxes = [_aabb(b) for b in self.bars]
        for i, b1 in enumerate(self.bars[:-1]):
            minx1, maxx1, minz1, maxz1 = boxes[i]
            for j, b2 in enumerate(self.bars[i + 1:], start=i + 1):
                minx2, maxx2, minz2, maxz2 = boxes[j]
                if (
                        maxx1 < minx2
                        or maxx2 < minx1
                        or maxz1 < minz2
                        or maxz2 < minz1
                ):
                    continue
                if segments_intersect(b1, b2):
                    raise ValueError(f"Bars {b1} and {b2} intersect.")

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
    def degree_of_static_indeterminacy(self):
        """Calculates the degree of static indeterminacy of the system.

        This function checks the static determinacy of the currently modified
        system. A system is statically determinate if n = 0.
        It is statically indeterminate if n > 0 and statically unstable
        if n < 0.

        Returns
        -------
        :any:`int`
            The number of redundant constraints in the system.
        """
        # self.logger.info(
        #     "Calculating the degree of static indeterminacy of "
        #     "the modified system.")

        support = sum((n.u != 'free') + (n.w != 'free') + (n.phi != 'free')
                      for n in self.nodes('bars'))
        # self.logger.debug(f"Total number of support reactions: {support}")

        hinge = sum(sum(h is True for h in b.hinge)
                    for b in self.bars)
        # self.logger.debug(f"Total number of hinges: {hinge}")

        n = support + 3 * len(self.bars) - (
                3 * len(self.nodes('bars')) + hinge)
        # self.logger.debug(f"Degree of static indeterminacy: {n}")

        return n

    @property
    def mesh(self):
        r"""Returns the generated mesh bars.

        Returns
        -------
        List[Bar | BarSecond]
            The list of bars representing the generated mesh.
        """
        return self._mesh.mesh

    @property
    def user_mesh(self):
        r"""Returns the user-defined mesh bars.

        Returns
        -------
        List[Bar | BarSecond]
            The list of bars resulting from user-defined divisions.
        """
        return self._mesh.user

    def mesh_segments_of(self, bar):
        r"""Returns the mesh segments corresponding to a given bar.

        Parameters
        ----------
        bar : Bar | BarSecond
            The original bar.

        Returns
        -------
        List[Bar | BarSecond]
            The list of mesh segments derived from the given bar.
        """
        return self._mesh.mesh_segments_of(bar)

    def user_segments_of(self, bar: Bar | BarSecond) -> List[Bar | BarSecond]:
        r"""Returns the user-defined segments of a given bar.

        Parameters
        ----------
        bar : Bar | BarSecond
            The original bar.

        Returns
        -------
        List[Bar | BarSecond]
            The list of user-defined segments of the bar.
        """
        return self._mesh.user_segments_of(bar)

    def bar_of(self, mesh_segment: Bar | BarSecond) -> Bar | BarSecond:
        r"""Returns the original bar of a given mesh segment.

        Parameters
        ----------
        mesh_segment : Bar | BarSecond
            A bar segment from the generated mesh.

        Returns
        -------
        Bar | BarSecond
            The original bar from which the mesh segment was created.
        """
        return self._mesh.bar_of(mesh_segment)

    def create_mesh(self, user_divisions=None):
        r"""Creates the mesh for the current system.

        Parameters
        ----------
        user_divisions : dict, optional
            A dictionary mapping bars to lists of relative positions at which
            the bars should be divided.

        Returns
        -------
        None

        Notes
        -----
        This method initializes and generates a :any:`Mesh` object based on
        the system bars and optional user-defined divisions.
        """
        self._mesh = Mesh(bars=self.bars, user_divisions=user_divisions)()


@dataclass(eq=False)
class Mesh:
    r"""Represents a mesh of bars.

    Attributes
    ----------
    bars : List[Bar | BarSecond]
        A list of bars to be meshed.
    user_divisions : Optional[Dict[Bar, List[:any:`float`]]]
        Optional user-defined divisions for each bar.

    Notes
    -----
    The `user_divisions` dictionary maps each bar to a list of positions
    where the bar should be divided.
    """
    bars: List[Bar | BarSecond]
    user_divisions: Optional[dict[Bar | BarSecond, list[float]]] = None

    def __post_init__(self):
        self.bars = list(self.bars)
        self._mesh: list[Bar | BarSecond] = []
        self._user: list[Bar | BarSecond] = []

        self._map_bar_user: dict[Bar | BarSecond, list[Bar | BarSecond]] = {}
        self._map_bar_mesh: dict[Bar | BarSecond, list[Bar | BarSecond]] = {}

    @property
    def mesh(self):
        r"""Gets the generated mesh.

        Returns
        -------
        List[Bar | BarSecond]
            The list of bars in the mesh.
        """
        return self._mesh

    @property
    def user(self):
        r"""Gets the user-defined mesh.

        Returns
        -------
        List[Bar | BarSecond]
            The list of bars in the user-defined mesh.
        """
        return self._user

    @property
    def _map_mesh_bar(self):
        r"""Gets a mapping from mesh bars to their original bars.

        Returns
        -------
        Dict[Bar | BarSecond, Bar | BarSecond]
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
                map_bar_mesh[bar] = [bar]
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

    def mesh_segments_of(self, bar: Bar | BarSecond) -> list[Bar | BarSecond]:
        r"""Gets the mesh segments corresponding to a given bar.

        Parameters
        ----------
        bar : Bar | BarSecond
            The bar for which to retrieve mesh segments.

        Returns
        -------
        List[Bar | BarSecond]
            The list of mesh segments corresponding to the given bar.
        """
        return self._map_bar_mesh.get(bar)

    def user_segments_of(self, bar: Bar | BarSecond) -> list[Bar | BarSecond]:
        r"""Gets the user-defined segments corresponding to a given bar.

        Parameters
        ----------
        bar : Bar | BarSecond
            The bar for which to retrieve user-defined segments.

        Returns
        -------
        List[Bar | BarSecond]
            The list of user-defined segments corresponding to the given bar.
        """
        return self._map_bar_user.get(bar)

    def bar_of(self, mesh_segment: Bar | BarSecond) -> Bar | BarSecond:
        r"""Gets the original bar corresponding to a given mesh segment.

        Parameters
        ----------
        mesh_segment : Bar | BarSecond
            The mesh segment for which to retrieve the original bar.

        Returns
        -------
        Bar | BarSecond
            The original bar corresponding to the given mesh segment.
        """
        return self._map_mesh_bar.get(mesh_segment)

    @staticmethod
    def _assign_loads_to_segments(load_pos_i, user_pos_i):
        r"""Assigns loads to segments based on their positions.

        Parameters
        ----------
        load_pos_i : Dict[:any:`float`, List[NodePointLoad]]
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
    def _interp_lloads(bar: Bar | BarSecond,
                       prev_bar: Optional[Bar | BarSecond],
                       pos: Optional[float] = None):
        r"""Interpolates line loads on a bar.

        Parameters
        ----------
        bar : Bar | BarSecond
            The bar for which to interpolate line loads.
        prev_bar : Optional[Bar | BarSecond]
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
               bar: Bar | BarSecond,
               pos_dict: Dict[float, List[NodePointLoad]]
               ) -> List[Bar | BarSecond]:
        r"""Splits a bar into segments based on user-defined positions and
        loads.

        Parameters
        ----------
        bar : Bar | BarSecond
            The bar to be split.
        pos_dict : Dict[float, List[NodePointLoad]]
            A dictionary of positions and loads.

        Returns
        -------
        List[Bar | BarSecond]
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
        bar : Bar | BarSecond
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
        bar : Bar | BarSecond
            The original bar.
        node_j : Node
            The end node of the segment.
        pos : float
            The position of the end node.
        end_point_loads : List[BarPointLoad]
            The point loads at the end node.

        Returns
        -------
        Bar | BarSecond
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
        bar : Bar | BarSecond
            The original bar.
        prev_bar : Bar | BarSecond
            The previous bar segment.
        node_j : Node
            The end node of the segment.
        pos : float
            The position of the end node.

        Returns
        -------
        Bar | BarSecond
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
        bar : Bar | BarSecond
            The original bar.
        prev_bar : Bar | BarSecond
            The previous bar segment.
        end_point_loads : List[BarPointLoad]
            The point loads at the end node.

        Returns
        -------
        Bar | BarSecond
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
