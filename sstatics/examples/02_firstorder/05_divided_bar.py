"""
Example 05:
Evaluate Bending Moments of a Load-Divided and User-Divided Bar

In this example, we analyze a simply supported beam using first-order
structural analysis.
A vertical point load is applied at the midspan of the beam, and the resulting
internal forces are evaluated.

- Automatic bar subdivision caused by loads acting inside a bar
- User-defined bar subdivision for result evaluation at specific locations
- Extraction of internal forces
- Plotting the bending moment distribution

This serves as a minimal introduction to the basic workflow of *sstatics*.
"""


# 1. Import required modules
from sstatics.core.preprocessing import (
    Bar, BarPointLoad, CrossSection, Polygon, Material, Node, System
)
from sstatics.core.calc_methods import FirstOrder
from sstatics.core.postprocessing.graphic_objects import (ObjectRenderer,
                                                          SystemGeo)

# 2. Define the cross-section using polygon geometry
cross_sec = CrossSection(
    geometry=[Polygon([(0, 0), (0.1, 0), (0.1, 0.2), (0, 0.2), (0, 0)])]
)

# 3. Define material (E = 11,000,000 kN/m²)
material = Material(11000000, 0.1, 0.1, 0.1)

# 4. Define nodes
n1 = Node(0, 0, u='fixed', w='fixed')
n2 = Node(3, 0, w='fixed')

# 5. Define the bar connecting the two nodes
bar_1 = Bar(n1, n2, cross_sec, material, point_loads=BarPointLoad(
    z=1, position=0.5))

# 6. Build the structural system
system = System([bar_1])

# Show system graphic
ObjectRenderer(SystemGeo(system), 'plotly').show()

# Create a system with user-defined subdivisions
system_user_division = System([bar_1])

# Add user-defined subdivision points
system_user_division.create_mesh({bar_1: [0.25, 0.75]})

# 7. Perform first-order structural analysis
solution = FirstOrder(system)
solution_user_division = FirstOrder(system_user_division)

# 8. Extract internal forces
forces = solution.internal_forces
forces_user_division = solution_user_division.internal_forces

print("Internal forces of the load-divided bars:\n", forces)
print("Internal forces of the user-divided bars:\n", forces_user_division)

# Note:
# It should be noted that the sign convention is determined according to the
# deformation method.

# 9. Plot bending moment distribution
solution.plot(kind='moment', mode='plotly')
solution_user_division.plot(kind='moment', mode='plotly')
