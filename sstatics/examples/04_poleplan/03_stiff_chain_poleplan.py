"""
Example 03:
Poleplan with a bending-stiff kinematic chain (partially movable system)

This example demonstrates the Poleplan for a complex multi-bar system.
Bar 1 forms an rigid chain because node_1 is fixed in all
degrees of freedom (u, w, phi), so this sub-chain does not move.
The rest of the system consists of kinematic chains that are free to move,
making the system partially movable.

The Poleplan shows the rigid-body displacements of all kinematic chains
and allows visualization of the rotation of individual chains.

A specific chain rotation (chain index 1) is manually set to 0.3
before plotting to illustrate the effect of the bending-stiff and fixed
elements on the mechanism behavior.
"""


from sstatics.core import Node, Bar, Material, CrossSection, System
from sstatics.core.solution import Poleplan
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)


# 1. Define material and cross-section
m_1 = Material(21000, 0.1, 8100, 0.1)          # S 235
c_1 = CrossSection(2769, 76.84, 20, 10, 0.1)   # HEA-240

# 2. Define nodes
node_1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
node_2 = Node(150, -400)
node_3 = Node(450, -400)
node_4 = Node(750, -600)
node_5 = Node(1250, -100, u='fixed', w='fixed')
node_6 = Node(1250, -500)
node_7 = Node(1250, -800, u='fixed', w='fixed')
node_8 = Node(1550, -100, w='fixed')

# 3. Define bars with hinges
bar_1 = Bar(node_1, node_2, c_1, m_1, hinge_phi_j=True)
bar_2 = Bar(node_2, node_3, c_1, m_1)
bar_3 = Bar(node_3, node_4, c_1, m_1, hinge_phi_i=True, hinge_phi_j=True)
bar_4 = Bar(node_4, node_5, c_1, m_1, hinge_phi_j=True)
bar_5 = Bar(node_5, node_6, c_1, m_1, hinge_phi_j=True)
bar_6 = Bar(node_4, node_6, c_1, m_1, hinge_phi_i=True, hinge_phi_j=True)
bar_7 = Bar(node_6, node_8, c_1, m_1, hinge_phi_i=True)
bar_8 = Bar(node_6, node_7, c_1, m_1)

bars = [bar_1, bar_2, bar_3, bar_4, bar_5, bar_6, bar_7, bar_8]

# 4. Define system
system = System(bars)

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 5. Create poleplan and plot rigid-body displacements
poleplan = Poleplan(system)

# -> Adjust rotation of a specific chain
poleplan.set_angle(chain_idx=1, angle=0.3)
poleplan.plot()

print("=== Poleplan Example with Bending-Stiff Chain ===")
print("Poleplan generated showing rigid-body displacements of chains "
      "including a bending-stiff element.")
