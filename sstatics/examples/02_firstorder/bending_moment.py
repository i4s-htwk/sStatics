
# Importing Modules
from sstatics.core.preprocessing import (
    Bar, CrossSection, Polygon, Material, Node, NodePointLoad, System
)
from sstatics.core.calc_methods import FirstOrder
from sstatics.core.postprocessing import SystemResult
from sstatics.graphic_objects import CrossSectionGraphic, ResultGraphic

# Define CrossSection by using coordinates
cross_sec = CrossSection(
    geometry=[Polygon([(0, 0), (0.1, 0), (0.1, 0.2), (0, 0.2), (0, 0)])]
)

# Visualize the cross-section
CrossSectionGraphic(cross_section=cross_sec).show()

# Define material: E-Modulus = 11,000,000 kN/m²
material = Material(11000000, 0.1, 0.1, 0.1)

# Define nodes
n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')   # Fixed support
n2 = Node(3, 0, loads=NodePointLoad(z=1))            # Loaded free end

# Define bar connecting Node 1 and Node 2
bar = Bar(n1, n2, cross_sec, material)

# Create system
system = System([bar])

# Perform calculation
solution = FirstOrder(system)

# Get bar deformations
forces = solution.internal_forces

# Show vertical displacement w at node 2
print('M at node 1 [kNm]: ', forces[0][2][0])

# Visualize Results

# Prepare results for plotting
results = SystemResult(
    system,
    solution.bar_deform_list,
    solution.internal_forces,
    solution.node_deform,
    solution.node_support_forces,
    solution.system_support_forces
)

# Plot vertical deflection
ResultGraphic(results, 'moment').show()
