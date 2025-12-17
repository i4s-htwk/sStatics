"""
Example 01:
Analytic Matrix Approach to calculate internal forces

This example demonstrates the second-order calculation using the
analytical matrix method.
The computation is performed on a cantilever column. The objective is to
determine the internal forces according to this approach and to plot the
bending-moment of the system.

This example can also be applied to the Taylor and P-Delta methods. In that
case, only Step 6 needs to be adjusted (use "taylor" or "p_delta" accordingly).
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

# 6. Set matrix approach and a chosen variant (here 'analytic')
sec_order.matrix_approach('analytic')

# 7. Get Solver object of the analytic matrix approach
solution_analytic = sec_order.solver_matrix_approach

# 8. Get the internal forces
# Index [0] for first bar
internal_forces_analytic = solution_analytic.internal_forces[0]

print("=== Second Order Example ===")
print(f"internal forces for analytical approach:\n {internal_forces_analytic}")
