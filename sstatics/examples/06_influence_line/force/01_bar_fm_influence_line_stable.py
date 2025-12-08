"""
Example 01:
Influence line for the force quantity fm

This example computes the influence line for the vertical force fm
at midspan of the first bar of a simple beam system.
In this case, the modified system remains stable and solvable using
the displacement method.

Therefore, the resulting influence line corresponds directly to the
bending line (deflection curve) of the solved modified system.
"""

from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System
)
from sstatics.core.calc_methods import InfluenceLine


# 1. Define material and cross-section
mat = Material(21000, 0.1, 8100, 0.1)          # S235
cs = CrossSection(2769, 76.84, 20, 10, 0.1)   # HEA-240

# 2. Define nodes with supports
node_1 = Node(0, 0, u='fixed', w='fixed')
node_2 = Node(300, 0, w='fixed')
node_3 = Node(600, 0, w='fixed')

# 3. Define bars and system
bar_1 = Bar(node_1, node_2, cs, mat)
bar_2 = Bar(node_2, node_3, cs, mat)
system = System([bar_1, bar_2])

# 4. Define Influence line module
il = InfluenceLine(system)

# 5. Influence line for vertical force fm in bar_1 at xi = 0.5
il.force('fm', bar_1, 0.5)

print("=== Influence Line Example ===")
print("Computed influence line for fm on bar_1 at xi = 0.5")
