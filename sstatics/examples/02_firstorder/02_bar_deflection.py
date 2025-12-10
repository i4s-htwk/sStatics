"""
Example 02:
Determining the vertical displacement at the free end of a cantilever beam

In this example, the same cantilever beam as in Example 01 is analyzed,
but the focus is shifted from internal forces to nodal displacements.
A point load acts at the free end, and the resulting deformation of the
beam is computed using first-order (linear) structural analysis.

This example demonstrates:

- Definition of a cross-section from polygon geometry
- Visualization of the cross-section
- Modeling a cantilever beam with a node load
- Extracting nodal deformations from the calculation result
- Reading the vertical displacement at the free end
- Plotting the displacement distribution

It provides a minimal introduction to displacement evaluation in *sstatics*.
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
n2 = Node(3, 0, loads=NodePointLoad(z=1))  # free end with vertical load

# 5. Define the bar connecting the two nodes
bar_1 = Bar(n1, n2, cross_sec, material)

# 6. Build the structural system
system = System([bar_1])

# 7. Perform first-order structural analysis
solution = FirstOrder(system)

# 8. Extract bar deformations
bar_deform = solution.bar_deform_total
print("Bar deformations of all bars:\n", bar_deform)

# Each entry is a 6×1 vector of bar deformations:
# [u'_i, w'_i, phi'_i, u'_j, w'_j, phi'_j]^T
bar_deform_1 = bar_deform[0]

# → The vertical bar displacement at the free end (node 2) is entry 5 (index 4)
w2 = bar_deform_1[4][0]
print("Vertical bar displacement w' at node 2 [m]:", w2)

# 9. Plot vertical displacement distribution
solution.plot(kind='w')
