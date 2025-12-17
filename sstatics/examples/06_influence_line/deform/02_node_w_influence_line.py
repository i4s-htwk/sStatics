"""
Example 02:
Influence line for the vertical displacement w

This example computes the influence line for the vertical displacement w
at node_3 of a simple beam system.

The calculation is based on the principle that for any sought displacement
(or rotation), a corresponding "unit force" is applied at that location.
Then, the displacement method is used to solve the system.
The resulting displacement curve of the system corresponds directly to
the influence line of the chosen displacement.
"""

from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System
)
from sstatics.core.calc_methods import InfluenceLine
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)

# 1. Define material and cross-section
mat = Material(21000, 0.1, 8100, 0.1)          # S235
cs = CrossSection(2769, 76.84, 20, 10, 0.1)   # HEA-240

# 2. Define nodes with supports
node_1 = Node(0, 0, u='fixed', w='fixed')
node_2 = Node(300, 0, w='fixed')
node_3 = Node(600, 0)

# 3. Define bars and system
bar_1 = Bar(node_1, node_2, cs, mat)
bar_2 = Bar(node_2, node_3, cs, mat)
system = System([bar_1, bar_2])

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 4. Define Influence line module
il = InfluenceLine(system)

# 5. Influence line for vertical displacement w at node_3
il.deform(kind='w', obj=node_3)

print("=== Influence Line Example ===")
print("Computed influence line for vertical displacement w at node_3")
