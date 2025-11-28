"""
Example 03:
P-Delta Matrix Approach to calculate bar deformations.

This example demonstrates the second-order calculation using the P-Delta matrix
method. The computation is performed on a cantilever column. The objective is
to determine the deformation in the w-direction at the free end of the bar in
the bar’s local coordinate system, and to plot the system deformation in the
w-direction.

This example can also be applied to the Analytic and Taylor methods. In that
case, only Step 6 needs to be adjusted (use "analytic" or "taylor"
accordingly).
"""

from sstatics.core.preprocessing import (Bar, BarLineLoad, CrossSection,
                                         Material, Node, NodePointLoad, System)
from sstatics.core.calc_methods import SecondOrder

# 1. Define cross-section and material -> steel and HEA 240 profile
c_1 = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
m_1 = Material(210000000, 0.1, 81000000, 0.1)

# 2. Define nodes (cantilever system)
node_1 = Node(x=0, z=0, u='fixed', w='fixed', phi='fixed')
node_2 = Node(x=0, z=-4, loads=NodePointLoad(x=0, z=182, phi=0, rotation=0))

# 3. Define bar
bar_1 = Bar(node_1, node_2, c_1, m_1, line_loads=BarLineLoad(
    pi=1, pj=1.5, direction='z', coord='bar', length='exact'))

# 4. Define system
system = System([bar_1])

# 5. Create a SecondOrder object
sec_order = SecondOrder(system)

# 6. Set matrix approach and a chosen variant (here 'p_delta')
sec_order.matrix_approach('p_delta')

# 7. Get Solver object of the p-delta matrix approach
solution_p_delta = sec_order.solver_matrix_approach

# 8. Get the bar deformations of the bar
# here only 1 bar -> index [0] of bar deform list
bar_deform = solution_p_delta.bar_deform_list[0]

# 9. Plot the deformation of the system
sec_order.plot('matrix', kind='w', decimals=4)

print("=== Second Order Example ===")
print(f"w_j = {bar_deform[4][0]} m")
