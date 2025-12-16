"""
Example 01:
Determining the bending moment at the fixed support of a cantilever beam

In this example, a simple cantilever beam is analyzed using first-order
structural analysis. A point load is applied at the free end, and the
resulting internal forces and support reactions are computed.
The example demonstrates:

- Definition of a cross-section from polygon geometry
- Modeling a beam with a fixed support and a free end load
- Extracting internal forces from the calculation result
- Reading the bending moment at the fixed support
- Accessing support reactions directly
- Plotting the bending moment distribution

This serves as a minimal introduction to the basic workflow of *sstatics*.
"""


# 1. Import required modules
from sstatics.core.preprocessing import (
    Bar, CrossSection, Polygon, Material, Node, NodePointLoad, System
)
from sstatics.core.calc_methods import FirstOrder

# 2. Define the cross-section using polygon geometry
cross_sec = CrossSection(
    geometry=[Polygon([(0, 0), (0.1, 0), (0.1, 0.2), (0, 0.2), (0, 0)])]
)

# 3. Define material (E = 11,000,000 kN/m²)
material = Material(11000000, 0.1, 0.1, 0.1)

# 4. Define nodes
n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')  # fully fixed support
n2 = Node(3, 0, loads=NodePointLoad(z=1))  # free end with load

# 5. Define the bar connecting the two nodes
bar_1 = Bar(n1, n2, cross_sec, material)

# 6. Build the structural system
system = System([bar_1])

# 7. Perform first-order structural analysis
solution = FirstOrder(system)

# 8. Extract internal forces of each bar
forces = solution.internal_forces
print("Internal forces of all bars:\n", forces)

# Each entry is a 6×1 vector: [f'x_i, f'z_i, f'm_i, f'x_j, f'z_j, f'm_j]^T
forces_bar_1 = forces[0]

# 9. Bending moment at node 1 (fixed end)
m_yy = forces_bar_1[2][0]  # index 2 = moment at bar start
print("Bending moment at node 1 [kNm]:", m_yy)

# Note:
# It should be noted that the sign convention is determined according to the
# deformation method.

# 10. Support reactions
support_forces = solution.node_support_forces
print("\nSupport forces:\n", support_forces)

# The vector has dimension (n_nodes_in_mesh * 3) × 1
# For node 1, the entries are:
#   [Px, Pz, Pm]^T  → M is at index 2
print("Fixed-end moment from support reaction:", support_forces[2][0])

# 11. Plot bending moment distribution
solution.plot(kind='moment')
