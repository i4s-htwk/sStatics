"""
Example 02:
Comparison of the internal forces between iteration steps.

This example demonstrates the second-order calculation using the
iterative approach.
The computation is performed on a cantilever column. The objective is to
determine the internal forces according to this approach between different
iteration steps and plot the bending moment of the inital iteration and the
last iteration according to the iterative approach.
"""

from sstatics.core.preprocessing import (Bar, BarLineLoad, CrossSection,
                                         Material, Node, NodePointLoad, System)
from sstatics.core.calc_methods import SecondOrder
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
node_1 = Node(x=0, z=0, u='fixed', w='fixed', phi='fixed')
node_2 = Node(x=0, z=-4, loads=NodePointLoad(x=0, z=182, phi=0, rotation=0))

# 3. Define bar
bar_1 = Bar(node_1, node_2, c_1, m_1, line_loads=BarLineLoad(
    pi=1, pj=1.5, direction='z', coord='bar', length='exact'))

# 4. Define system
system = System([bar_1])

# 5. Create a SecondOrder object
sec_order = SecondOrder(system)

# 6. Set iterative approach and chose the number of iteration steps,
# the iteration tolerance and what kind of result you are looking for
iteration_steps = 20
tolerance = 10e-4
sec_order.iterative_approach(iterations=iteration_steps, tolerance=tolerance)

# 7. Get Solver object of the iterative approach
solution_inital_iteration = sec_order.solver_iteration_cumulative(0)
solution_last_iteration = sec_order.solver_iteration_cumulative(-1)

# 8. Get the internal forces
# Index [0] for first bar
internal_forces_inital_iteration = solution_inital_iteration.internal_forces[0]
internal_forces_last_iteration = solution_last_iteration.internal_forces[0]

# 9. Plot of the bending moments
sec_order.plot('iterative', 0, kind='moment')
sec_order.plot('iterative', -1, kind='moment')

print("=== Second Order Example ===")
print(f"Internal forces for the inital iteration:\n "
      f"{internal_forces_inital_iteration}")
print(f"Internal forces for the last iteration:\n "
      f"{internal_forces_last_iteration}")
