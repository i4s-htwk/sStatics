
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.modifier import SystemModifier
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.poleplan.operation import get_angle
from sstatics.core.preprocessing.system import System
from sstatics.core.solution.first_order import FirstOrder


@dataclass(eq=False)
class InfluenceLine:

    system: System

    def __post_init__(self):
        self.dof = 3
        self.modifier = SystemModifier(self.system)

    def force(self, force: Literal['fx', 'fz', 'fm'], obj,
              position: float = 0):
        if force not in ['fx', 'fz', 'fm']:
            raise ValueError(f"Invalid force type: {force}")

        if isinstance(obj, Bar):
            self.modified_system = self.modifier.modify_bar_force_influ(
                obj, force, position, virt_force=1)

        elif isinstance(obj, Node):
            if position:
                raise ValueError(
                    "If obj is an instance of Node, position must be None.")
            self.modified_system = self.modifier.modify_node_force_influ(
                obj, force, virt_force=1)
        else:
            raise ValueError("obj must be an instance of Bar or Node")

        calc_system = FirstOrder(self.modified_system)

        if calc_system.solvable:
            norm_force = self.calc_norm_force(force, obj)
            deform_1, force_1 = calc_system.calc

            deform_2 = [vec * norm_force for vec in deform_1]
            force_2 = [vec * norm_force for vec in force_1]
            return deform_2, force_2
        else:
            # 1. polplan aufstellen
            # self.modified_system.get_polplan()
            # TODO: --> so war es vorher, ich glaube das braucht man nicht mehr

            if self.modified_system.polplan.solved:
                # 2. Winkelberechnung für Scheibe in dem das obj enthalten ist
                chain, angle = self.get_chain_and_angle(obj, force, position)

                # 2.3 Berechnung aller weiteren Winkel
                self.modified_system.polplan.set_angle(
                    target_chain=chain, target_angle=angle)

                # 3. Berechnung der Verschiebungsfigur
                return (self.modified_system.polplan.get_displacement_figure(),
                        None)
            else:
                raise ValueError('poleplan is not solved.')

    def get_chain_and_angle(self, obj, force, position):
        if isinstance(obj, Bar):
            angle = 0
            # 2.1 Um welche Scheibe handelt es sich
            idx = list(self.system.bars).index(obj)
            bar = self.modified_system.bars[idx]
            chain = self.modified_system.polplan.get_chain(bars={bar})

            # 2.2 Wie groß ist der Winkel
            if force == 'fz':
                if position in {0, 1}:
                    node = obj.node_i if position == 0 else obj.node_j
                    displacement = 1 if position == 0 else -1

                    if chain.absolute_pole.is_infinite:
                        print(
                            f'Position {position}: {chain.absolute_pole} '
                            f'liegt im Unendlichen!')
                        aPole_coords, node_coords, c = (
                            self.modified_system.polplan.find_adjacent_chain(
                                node, chain))

                        if aPole_coords is None:
                            print('Schauen, ob es angrenzende Scheiben gibt')
                            for rPole in chain.relative_pole:
                                if rPole != node:
                                    aPole_coords, node_coords, c = (
                                        self.modified_system.polplan.
                                        find_adjacent_chain(
                                            rPole.node, chain))
                        idx_chain = self.modified_system.polplan.chains.index(
                            chain)
                        next_chain = self.modified_system.polplan.chains.index(
                            c)

                        angle = get_angle(point=node_coords,
                                          center=aPole_coords,
                                          displacement=displacement)
                        if idx_chain < next_chain:
                            angle = angle / c.angle_factor
                    else:
                        print('aPol liegt nicht im Unendlichen!')
                        aPole_coords = chain.absolute_pole.coords
                        node_coords = np.array([[node.x],
                                                [node.z]])

                        angle = get_angle(point=node_coords,
                                          center=aPole_coords,
                                          displacement=displacement)
                else:
                    print(bar.node_j)
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
            # 2.1 Um welche Scheibe handelt es sich
            chain = self.modified_system.polplan.get_chain_node(obj)
            angle = 0

            # 2.2 Wie groß ist der Winkel
            if force == 'fz':
                aPole_coords = chain.absolute_pole.coords
                node_coords = np.array([[obj.x], [obj.z]])

                angle = get_angle(point=node_coords,
                                  center=aPole_coords,
                                  displacement=1)
            elif force == 'fm':
                angle = -1

            return chain, angle

    def calc_norm_force(self, force: Literal['fx', 'fz', 'fm'],
                        obj):
        """
        Normalize the deformation of the bar system based on the given force.
        This method calculates a virtual force to balance the deformation
        difference between two connected bars, based on their deformation.
        """
        calc_system = FirstOrder(self.modified_system)
        if isinstance(obj, Bar):
            # calc bar deformations
            deform = calc_system.bar_deform_list

            # Get the index of the bar in the system
            bars = list(self.system.bars)
            idx = bars.index(obj)

            deform_bar_i, deform_bar_j = deform[idx], deform[idx + 1]

            # Map the force type to corresponding indices for the deformation
            # values
            force_indices = {'fx': (3, 0), 'fz': (4, 1), 'fm': (5, 2)}
            idx_i, idx_j = force_indices[force]

            # Calculate the difference in deformation between the two bars
            delta = deform_bar_j[idx_j][0] - deform_bar_i[idx_i][0]
        elif isinstance(obj, Node):
            node_deformation = calc_system.node_deform
            for i, node in enumerate(self.system.nodes()):
                if node == obj:
                    node_deform = node_deformation[
                               i * self.dof:i * self.dof + self.dof, :]
                    force_indices = {'fx': 0, 'fz': 1, 'fm': 2}
                    delta = node_deform[force_indices[force]][0]
                    break
        else:
            raise ValueError("obj must be an instance of Bar or Node")

        if delta == 0:
            raise ZeroDivisionError("Deformation difference (delta) is zero, "
                                    "cannot calculate norm force.")
        return -1 * float(np.abs(1 / delta))

    def deform(self, deform: Literal['u', 'w', 'phi'], obj,
               position: float = 0):
        if deform not in ['u', 'w', 'phi']:
            raise ValueError(f"Invalid deform type: {deform}")

        if isinstance(obj, Bar):
            if not (0 <= position <= 1):
                raise ValueError(
                    f"Position {position} must be between 0 and 1.")

            self.modified_system = (
                self.modifier.modify_bar_deform_influ(obj, deform, position))

        elif isinstance(obj, Node):
            if position:
                raise ValueError(
                    "If obj is an instance of Node, position must be None.")
            self.modified_system = self.modifier.modify_node_deform(
                obj, deform, virt_force=1)
        else:
            raise ValueError("obj must be an instance of Bar or Node")

        calc_system = FirstOrder(self.modified_system)

        return calc_system.calc
