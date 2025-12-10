"""
Example 03:
Computing the horizontal displacement of a spring-supported node using
the Principle of Virtual Forces (PVK)

In this example, node_5 is supported by a translational spring. The spring
stiffness contributes to the work equation of the node, and the horizontal
displacement u of node_5 is computed by applying a virtual force in the
x-direction.

The system includes bars subjected to line loads and point loads. Point
loads on bars automatically create additional nodes and bar segments in
the computational mesh to correctly account for their effects. The PVF
method evaluates the work contributions of all bars and nodes, including
spring-supported nodes, to determine the target displacement.

Additionally, the example demonstrates how to inspect:
1. The work matrices for all nodes and bars.
2. The specific contribution of an individual node or bar (including bar
   segments) using the `work_of(...)` method.
This allows detailed insight into how individual elements contribute
to the overall deformation.
"""

# 1. Import modules
from sstatics.core.preprocessing import (
    Bar, BarLineLoad, BarPointLoad, CrossSection, Material, Node,
    NodePointLoad, System
)
from sstatics.core.calc_methods import PVF

# 2. Define material
s253 = Material(210000000, 0.1, 81000000, 0.1)

# 3. Define cross-sections
hea_220 = CrossSection(5410 * 1e-8, 33.4 * 1e-4, 0.240, 0.120, 5/6)
heb_220 = CrossSection(8090 * 1e-8, 33.4 * 1e-4, 0.240, 0.120, 5/6)

# 4. Define nodes
n1 = Node(0, 0)
n2 = Node(4, 0)
n3 = Node(4, 3*1.5, u='fixed', w='fixed')
n4 = Node(8, 0, loads=NodePointLoad(z=4))
n5 = Node(14, 3*1.5, w=500)

# 5. Define loads and deformation components
line_load = BarLineLoad(pi=6, pj=6)
bar_point_load_1 = BarPointLoad(x=6, position=1/3)
bar_point_load_2 = BarPointLoad(x=6, position=2/3)
def_comp = 'moment'

# 6. Define bars
b1 = Bar(n1, n2, hea_220, s253,
         line_loads=line_load,
         deformations=def_comp)
b2 = Bar(n2, n4, hea_220, s253,
         line_loads=line_load,
         deformations=def_comp)
b3 = Bar(n2, n3, heb_220, s253,
         point_loads=[bar_point_load_1, bar_point_load_2],
         deformations=def_comp)
b4 = Bar(n4, n5, heb_220, s253,
         deformations=def_comp)

# 7. Assemble system
system = System(bars=[b1, b2, b3, b4])

# 8. PVK calculation
pvf = PVF(system=system)

# Apply a virtual force fx at node 5 to obtain x(node 5)
pvf.add_virtual_node_load(n5, 'fx')

# 8. Compute horizontal displacement of node_5 using PVF
pvf = PVF(system)

# Apply a virtual horizontal force fx at node_5
pvf.add_virtual_node_load(n5, 'fx')

node_5_u = pvf.deformation()
print("Horizontal displacement at node 5 (PVK) [m]:", node_5_u)

# 9. Inspect work contributions
# The work matrices show how all nodes and bars contribute to the total work
# equation.
# Note: point loads on bars introduce additional nodes and bar segments in
# the computational mesh. The system now contains 6 bars and 7 nodes.

# Work contributions of all nodes
work_matrix_nodes = pvf.work_matrix(kind='nodes')
print("Work matrix (nodes):\n", work_matrix_nodes)

# The original model node 5 is mapped to position 7 in the mesh.
# Use `work_of(...)` to get its specific contribution in the PVF analysis.
work_n5 = pvf.work_of(obj=n5)
print("Work contribution of node 5:\n", work_n5)

# Work contributions of all bars
work_matrix_bars = pvf.work_matrix(kind='bars')
print("Work matrix (bars):\n", work_matrix_bars)

# For bars split by point loads (e.g., bar 3), you can inspect individual
# segment contributions
work_b3_seg = pvf.work_of(obj=b3, sum=False)
print("Work contributions of bar 3 segments:\n", work_b3_seg)

# Or get the total contribution of all segments combined
work_b3 = pvf.work_of(obj=b3, sum=True)
print("Total work contribution of bar 3:\n", work_b3)
