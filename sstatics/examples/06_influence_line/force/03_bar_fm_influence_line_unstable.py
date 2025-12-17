"""
Example 03:
Influence line for the force quantity fm (mechanism case)

This example computes the influence line for the force quantity fm
at midspan of the first bar of a simple beam system.
In this case, the modified system becomes kinematically admissible (a
mechanism) and cannot be solved with the displacement method.

Because the modified system is a mechanism, the influence line equals
the displacement figure of the corresponding rigid-body motion.
To obtain this figure, a poleplan is constructed first, followed
by evaluation of the resulting displacement shape.
"""

from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System
)
from sstatics.core.calc_methods import InfluenceLine
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)


# 1. Define material and cross-section
m_1 = Material(21000, 0.1, 8100, 0.1)          # S235
c_1 = CrossSection(2769, 76.84, 20, 10, 0.1)   # HEA-240

# 2. Define nodes with supports
node_1 = Node(0, 0, u='fixed', w='fixed')
node_2 = Node(300, 0, w='fixed')
node_3 = Node(600, 0)

# 3. Define bars and system
bar_1 = Bar(node_1, node_2, c_1, m_1)
bar_2 = Bar(node_2, node_3, c_1, m_1)
system = System([bar_1, bar_2])

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 4. Define Influence line module
il = InfluenceLine(system)

# 5. Influence line for vertical force fm in bar_1 at xi = 0.5
il.force('fm', bar_1, 0.5)

il.plot(scale=10)

print("=== Influence Line Example ===")
print("Computed influence line for fm on bar_1 at xi = 0.5")
