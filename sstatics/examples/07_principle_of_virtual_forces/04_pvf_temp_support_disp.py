"""
Example 04:
Computing the vertical displacement w at a node using the PVF with
temperature loads and prescribed support displacements

This example demonstrates the use of the Principle of Virtual Forces (PVF)
to compute the vertical displacement w at node_2 (x = 2.5 m) in a mixed
structural system influenced by temperature loads and prescribed support
displacements.

All three internal force contributions are considered in the deformation:
- bending moment
- axial force
- shear force
(deformations = ('moment', 'normal', 'shear'))

Two special aspects distinguish this example:

1. **Temperature loads:**
   Each bar may expand or contract due to a defined temperature distribution.
   These thermal strains generate additional internal forces that contribute
   to the final displacement.

2. **Support displacement at node_6:**
   A prescribed vertical displacement is imposed, introducing an additional
   deformation source that is fully accounted for by the PVF.

The combined effects of structural loading, temperature-induced strains, and
support movements are captured by applying a virtual vertical unit load at
node_2. The PVF method integrates the internal virtual work across all bars
to determine the resulting displacement w(n2).
"""

# 1. Import required modules
from sstatics.core.calc_methods import PVF
from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System, BarLineLoad, BarPointLoad,
    BarTemp, NodeDisplacement
)
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)

# 2. Define material
mat_1 = Material(210000000, 1, 81000000, 0.000012)

# 3. Define cross-sections
cs_1 = CrossSection(0.0000277, 0.00334, 0.22, 0.11, 2.807)
cs_2 = CrossSection(0.0000579, 0.00459, 0.27, 0.135, 2.782)
cs_3 = CrossSection(0.0000369, 0.00538, 0.19, 0.2, 4.847)
cs_4 = CrossSection(0.0000167, 0.00388, 0.152, 0.16, 4.826)

# 4. Define nodes
n1 = Node(0, 0, u="free",  w="fixed")
n2 = Node(2.5, 0)
n3 = Node(4, 0)
n4 = Node(8, 0)
n5 = Node(9.5, 0)
n6 = Node(4, 3.5, u="fixed", w="fixed", displacements=NodeDisplacement(z=0.02))
n7 = Node(8, 3.5, u="fixed", w="fixed")

# 5. Define loads and deformation components
bar_line_load = BarLineLoad(pi=20, pj=20, direction="z")
bar_point_load = BarPointLoad(30, position=1.5 / 3.5)
bar_temp_1 = BarTemp(30, 10)
bar_temp_2 = BarTemp(30, 30)
bar_temp_3 = BarTemp(10, 10)
bar_temp_4 = BarTemp(10, 30)
def_comp = ('moment', 'normal', 'shear')

# 6. Define bars
b1 = Bar(n1, n2, cs_1, mat_1, hinge_phi_j=True, line_loads=[bar_line_load],
         temp=bar_temp_1, deformations=def_comp)
b2 = Bar(n2, n3, cs_2, mat_1, line_loads=[bar_line_load], temp=bar_temp_1,
         deformations=def_comp)
b3 = Bar(n3, n4, cs_2, mat_1, line_loads=[bar_line_load], temp=bar_temp_1,
         deformations=def_comp)
b4 = Bar(n4, n5, cs_2, mat_1, line_loads=[bar_line_load], temp=bar_temp_2,
         deformations=def_comp)
b5 = Bar(n6, n3, cs_3, mat_1, point_loads=[bar_point_load], temp=bar_temp_3,
         deformations=def_comp)
b6 = Bar(n7, n4, cs_4, mat_1, hinge_phi_j=True, temp=bar_temp_4,
         deformations=def_comp)

# 7. Assemble system
system = System([b1, b2, b3, b4, b5, b6])

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 8. PVF calculation
pvf = PVF(system)

# Apply a virtual vertical force fz at node_2 to obtain w(n2)
pvf.add_virtual_node_load(n2, force="fz")

# Show virtual system
ObjectRenderer(SystemGeo(pvf.virtual_system), 'plotly').show()

# Compute and print vertical displacement at node_2
w_n2 = pvf.deformation()
print("Vertical displacement at node 2 (PVK) [m]:", w_n2)

# 9. Plot moment distribution
pvf.plot('real', kind='moment', mode='plotly')
pvf.plot('virt', kind='moment', mode='plotly')
