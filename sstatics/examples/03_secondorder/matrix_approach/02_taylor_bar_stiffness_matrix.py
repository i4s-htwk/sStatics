"""
Example 02:
Representation of the Modified Stiffness Matrix for the Taylor Series Approach

This example demonstrates the second-order calculation using the Taylor matrix
method. The computation is performed on a cantilever column. The objective is
to examine the modified stiffness matrix derived from the Taylor series
expansion for second-order theory and compare it with the element stiffness
matrix according to first-order theory.

This example can also be applied to the Analytical and P-Delta methods. In that
case, only Step 6 needs to be adjusted (use "analytic" or "p_delta"
accordingly).
"""

from sstatics.core.preprocessing import (Bar, BarLineLoad, CrossSection,
                                         Material, Node, NodePointLoad, System)
from sstatics.core.calc_methods import SecondOrder
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)

# Optional: Modify the output of the print
import numpy as np
np.set_printoptions(
    precision=4,      # number of decimals
    suppress=True,    # suppress scientific notation for small numbers
)

# 1. Define cross-section and material -> steel and HEA 240 profile
c_1 = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
m_1 = Material(210000000, 0.1, 81000000, 0.1)

# 2. Define nodes (cantilever system)
node_1 = Node(x=0, z=0, u='fixed', w='fixed', phi='fixed', rotation=np.pi/2)
node_2 = Node(x=0, z=-4, loads=NodePointLoad(x=0, z=182, phi=0, rotation=0))

# 3. Define bar
bar_1 = Bar(node_1, node_2, c_1, m_1, line_loads=BarLineLoad(
    pi=1, pj=1.5, direction='z', coord='bar', length='exact'))

# 4. Define system
system = System([bar_1])

# Show system graphic
ObjectRenderer(SystemGeo(system), 'plotly').show()

# 5. Create a SecondOrder object
sec_order = SecondOrder(system)

# 6. Set matrix approach and a chosen variant (here 'taylor')
sec_order.matrix_approach('taylor')

# 7. Get modified system for the matrix approach for variant taylor
second_order_system = sec_order._modified_system_matrix

# 8. Get first bar of the system
bar_second = second_order_system.bars[0]

# 9. Show modified stiffness matrix for the taylor approach
stiffness_matrix_modified = bar_second.stiffness_matrix()

# 10. Compare original stiffness matrix and modified stiffness matrix
stiffness_matrix_original = bar_1.stiffness_matrix()

print("=== Second Order analytic matrix approach Example ===")
print(f"Original stiffness matrix:\n {stiffness_matrix_original}")
print(f"Modified stiffness matrix:\n {stiffness_matrix_modified}")
