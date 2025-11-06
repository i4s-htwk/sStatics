
from dataclasses import dataclass, field
from typing import Literal, List

import numpy as np

from sstatics.core.logger_mixin import LoggerMixin
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.poleplan import Poleplan
from sstatics.core.preprocessing.poleplan.operation import get_angle
from sstatics.core.preprocessing.system import System
from sstatics.core.solution.first_order import FirstOrder
from sstatics.core.postprocessing import BarResult, RigidBodyDisplacement


@dataclass(eq=False)
class InfluenceLine(LoggerMixin):

    system: System
    debug: bool = False

    _modified_system: System | None = field(init=False, default=None)

    _solution: FirstOrder | None = field(init=False, default=None)
    _norm_force: float | None = field(init=False, default=None)
    _deflections: List[BarResult] | None = field(init=False, default=None)

    _poleplan: Poleplan | None = field(init=False, default=None)
    _rigid_motions: List[RigidBodyDisplacement] | None = (
        field(init=False, default=None))

    def force(
            self,
            kind: Literal['fx', 'fz', 'fm'],
            obj: Bar | Node,
            position: float = 0.0,
            n_disc: int = 10
    ):
        self._reset_results()

        if kind not in ('fx', 'fz', 'fm'):
            raise ValueError(f"Invalid kind type: {kind}")

        self._modified_system = self._modify_system(obj, kind, position)
        self._solution = FirstOrder(system=self._modified_system)

        if self.solution.solvable:
            self._norm_force = self._compute_norm_force(kind, obj)
            if self._norm_force != 0:
                self._modified_system = self._modify_system(
                    obj, kind, position, virt_force=self.norm_force
                )
                self._solution = FirstOrder(system=self._modified_system)
            self._create_deflection_objects(n_disc=n_disc)

        else:
            self._poleplan = Poleplan(system=self.modified_system,
                                      debug=self.debug)
            chain, angle = self._compute_chain_angle(obj, kind, position)
            self._poleplan.set_angle(target_chain=chain, target_angle=angle)
            self._rigid_motions = self._poleplan.rigid_motion(n_disc=n_disc)

        self.plot()

    def deform(self,
               kind: Literal['u', 'w', 'phi'],
               obj,
               position: float = 0,
               n_disc: int = 10):
        self._reset_results()

        if kind not in ['u', 'w', 'phi']:
            raise ValueError(f"Invalid kind type: {kind}")

        self._modified_system = self._modify_system(obj, kind, position)
        self._solution = FirstOrder(system=self._modified_system)

        self._create_deflection_objects(n_disc=n_disc)

        self.plot()

    @property
    def modified_system(self) -> System:
        if self._modified_system is None:
            raise AttributeError(
                "The modified system has not been created yet. "
                "Call `force()` or `deform()` before accessing "
                "`modified_system`."
            )
        return self._modified_system

    @property
    def solution(self) -> FirstOrder:
        if self._solution is None:
            raise AttributeError(
                "No solution is available. "
                "Call `force()` or `deform()` before accessing `solution`."
            )
        return self._solution

    @property
    def norm_force(self) -> float:
        if self._norm_force is None:
            raise AttributeError(
                "The normalizing virtual force is not defined. "
                "Call `force()` before accessing `norm_force`."
            )
        return self._norm_force

    @property
    def poleplan(self) -> Poleplan:
        if self._poleplan is None:
            # Case 1: influence line solved via deformation method
            if self._solution is not None:
                raise ValueError(
                    "No pole plan is available because the influence line "
                    "was solved by the deformation method. "
                    "A pole plan is only used when the system becomes a "
                    "mechanism."
                )
            # Case 2: no influence line has been computed yet
            raise AttributeError(
                "No pole plan has been computed yet. "
                "Call `force()` before accessing `poleplan`."
            )
        return self._poleplan

    @property
    def deflections(self) -> List[BarResult]:
        if self._solution is None:
            raise AttributeError(
                "No deflection data available. "
                "Call `force()` or `deform()` before accessing `deflections`."
            )
        if self._deflections is None and not self.solution.solvable:
            raise AttributeError(
                "No deflection data available because the system became "
                "a mechanism after releasing the constraint. "
                "In this case, the influence line corresponds to the "
                "rigid-body displacement of the kinematic chain and can be "
                "accessed via `rigid_motions`."
            )
        return self._deflections

    @property
    def rigid_motions(self) -> List[RigidBodyDisplacement]:
        if self._rigid_motions is None:
            raise AttributeError(
                "No rigid-body motion data available. "
                "Call `force()` before accessing `rigid_motions`. "
                "Rigid-body motions are only generated when the system "
                "becomes a mechanism."
            )
        return self._rigid_motions

    def _modify_system(
            self,
            obj: Bar | Node,
            kind,
            position: float,
            virt_force: float = 1) -> System:

        modifier = SystemModifier(self.system)

        is_bar = isinstance(obj, Bar)
        is_node = isinstance(obj, Node)

        if not (is_bar or is_node):
            raise TypeError("obj must be an instance of Bar or Node.")

        if is_node and position != 0:
            raise ValueError("If obj is a Node, `position` must be 0.")

        force_kinds = {'fx', 'fz', 'fm'}
        deform_kinds = {'u', 'w', 'phi'}

        if kind in force_kinds:
            if is_bar:
                return modifier.modify_bar_force_influ(obj, kind, position,
                                                       virt_force)
            return modifier.modify_node_force_influ(obj, kind, virt_force)

        elif kind in deform_kinds:
            if is_bar:
                return modifier.modify_bar_deform_influ(obj, kind, position)
            return modifier.modify_node_deform(obj, kind, position)

        else:
            raise ValueError(
                f"Invalid kind '{kind}'. Must be one of "
                f"{force_kinds | deform_kinds}.")

    def _compute_norm_force(self, force: Literal['fx', 'fz', 'fm'],
                            obj: Bar | Node) -> float:
        """Berechnet die normierende virtuelle Kraft."""
        delta = 0.0
        if isinstance(obj, Bar):
            deform = self.solution.bar_deform_list
            bars = list(self.system.bars)
            idx = bars.index(obj)

            d_i, d_j = deform[idx], deform[idx + 1]
            mapping = {'fx': (3, 0), 'fz': (4, 1), 'fm': (5, 2)}
            i, j = mapping[force]

            delta = d_j[j][0] - d_i[i][0]
        elif isinstance(obj, Node):
            node_deform = self.solution.node_deform
            for i, n in enumerate(self.system.nodes()):
                if n is obj:
                    slice_ = slice(i * 3, i * 3 + 3)
                    nd = node_deform[slice_, :]
                    mapping = {'fx': 0, 'fz': 1, 'fm': 2}
                    delta = nd[mapping[force]][0]
                    break
        else:
            raise TypeError("obj must be an instance of Bar or Node")
        if delta == 0:
            raise ZeroDivisionError(
                "Delta is zero – cannot compute norm force.")
        return -float(np.abs(1 / delta))

    def _compute_chain_angle(self, obj, force, position):
        angle = 0
        if isinstance(obj, Bar):
            idx = list(self.system.bars).index(obj)
            bar = self.modified_system.bars[idx]
            chain = self._poleplan.get_chain(bars={bar})

            if force == 'fz':
                if position in {0, 1}:
                    node = obj.node_i if position == 0 else obj.node_j
                    displacement = 1 if position == 0 else -1

                    if chain.absolute_pole.is_infinite:
                        aPole_coords, node_coords, c = (
                            self._poleplan.find_adjacent_chain(node, chain)
                        )
                        if aPole_coords is None:
                            for rPole in chain.relative_pole:
                                if rPole != node:
                                    aPole_coords, node_coords, c = (
                                        self._poleplan.find_adjacent_chain(
                                            rPole.node, chain)
                                    )
                        idx_chain = self._poleplan.chains.index(chain)
                        next_chain = self._poleplan.chains.index(c)

                        angle = get_angle(point=node_coords,
                                          center=aPole_coords,
                                          displacement=displacement)
                        if idx_chain < next_chain:
                            angle = angle / c.angle_factor
                    else:
                        aPole_coords = chain.absolute_pole.coords
                        node_coords = np.array([[node.x], [node.z]])

                        angle = get_angle(point=node_coords,
                                          center=aPole_coords,
                                          displacement=displacement)
                else:
                    angle = -1 / obj.length
            elif force == 'fm':
                if position == 0:
                    angle = -1
                elif position == 1:
                    angle = 1
                else:
                    angle = (1 - position) / obj.length

            return chain, angle
        elif isinstance(obj, Node):
            chain = self._poleplan.get_chain_node(obj)

            if force == 'fz':
                aPole_coords = chain.absolute_pole.coords
                node_coords = np.array([[obj.x], [obj.z]])

                angle = get_angle(point=node_coords,
                                  center=aPole_coords,
                                  displacement=1)
            elif force == 'fm':
                angle = -1

            return chain, angle

    def _create_deflection_objects(self, n_disc=10):
        deflections = []
        for i, bar in enumerate(self.modified_system.mesh):
            dgl = BarResult(
                bar=bar,
                forces=self.solution.internal_forces[i],
                deform=self.solution.bar_deform_list[i],
                n_disc=n_disc
            )
            deflections.append(dgl)
        self._deflections = deflections

    def _reset_results(self):
        """Reset all computed attributes to None before a new calculation."""
        self._modified_system = None
        self._solution = None
        self._norm_force = None
        self._deflections = None
        self._poleplan = None
        self._rigid_motions = None

    def plot(self, mode: str = 'MPL'):
        """Plot the influence line, either from deformation or rigid-body
        motion results."""
        # geo_system = GeoSystem(self.system)
        #
        # if self.deflections is not None:
        #     result = self.deflections
        # elif self.rigid_motions is not None:
        #     result = self.rigid_motions
        # else:
        #     raise AttributeError(
        #         "No influence line data found. "
        #         "Call `force()` or `deform()` before using `plot()`."
        #     )
        #
        # geo_result_line = GeoResultLine(self.system.mesh, result=result)
        # Renderer([geo_system, geo_result_line], mode).show()

        if self._deflections is not None:
            from sstatics.graphic_objects import ResultGraphic
            from sstatics.core.postprocessing import SystemResult
            sol = self.solution
            deform = sol.bar_deform_list
            forces = sol.internal_forces
            node_def = sol.node_deform
            node_supp = sol.node_support_forces
            sys_supp = sol.system_support_forces

            result = SystemResult(
                system=self.modified_system,
                bar_deform_list=deform,
                bar_internal_forces=forces,
                node_deform=node_def,
                node_support_forces=node_supp,
                system_support_forces=sys_supp,
            )

            result.bars = self.deflections

            ResultGraphic(system_result=result, kind='w').show()
        elif self._rigid_motions is not None:
            from sstatics.graphic_objects.poleplan import PoleplanGraphic
            PoleplanGraphic(poleplan=self.poleplan).show()
        else:
            print(mode)
            raise AttributeError(
                "No influence line data found. "
                "Call `force()` or `deform()` before using `plot()`."
            )
