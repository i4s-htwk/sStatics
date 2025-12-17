"""
Example 03:
Computing a deformation in a large statically indeterminate frame
using the Reduction Theorem

In this example, a large statically indeterminate frame is analyzed. The
system has multiple redundants, and the Reduction Theorem is used to remove
them by releasing specific degrees of freedom (hinges or node releases),
forming a statically determinate system.

After removing the redundant constraints, the deformation at a specific
location is computed directly using the Reduction Theorem. A hand-calculated
reference value using integration formulas is also provided to validate
the numerical result.

This example demonstrates the application of the Reduction Theorem in
complex frames with high degrees of static indeterminacy, without
introducing additional virtual loads.
"""

# 1. Import required modules
from sstatics.core.calc_methods import ReductionTheorem
from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System, BarLineLoad, Polygon
)
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)
import numpy as np

# 2. Define material
material = Material(3300e4, 1, 1, 1)

# 3. Define cross-sections using polygon geometry
rect_1 = Polygon(points=[(0, 0), (0.35, 0), (0.35, 0.35), (0, 0.35), (0, 0)])
rect_2 = Polygon(points=[(0, 0), (0.35, 0), (0.35, 0.6), (0, 0.6), (0, 0)])

cs_1 = CrossSection(geometry=[rect_1])
cs_2 = CrossSection(geometry=[rect_2])

# 4. Define nodes
node_a = Node(0, 3, u='fixed', w='fixed', phi='fixed')
node_b = Node(18, 5, u='fixed', w='fixed', phi='fixed')
node_c = Node(-2, 0, u='fixed', w='fixed')
node_d = Node(4, 0)
node_e = Node(12, 0)
node_f = Node(22, 0, u='fixed', w='fixed')

# 5. Define bar loads
line_load_1 = BarLineLoad(pi=28, pj=28, direction='z')
line_load_2 = BarLineLoad(pi=2.6, pj=2.6, direction='x', coord='system',
                          length='proj')
line_load_3 = BarLineLoad(pi=-10, pj=-10, direction='x', coord='system',
                          length='proj')

# 6. Define bars
bar_1 = Bar(node_a, node_d, cs_1, material, line_loads=line_load_2,
            deformations="moment")
bar_2 = Bar(node_e, node_b, cs_1, material, line_loads=line_load_3,
            deformations="moment")
bar_3 = Bar(node_c, node_d, cs_2, material, line_loads=line_load_1,
            deformations="moment")
bar_4 = Bar(node_d, node_e, cs_2, material, line_loads=line_load_1,
            deformations="moment")
bar_5 = Bar(node_e, node_f, cs_2, material, line_loads=line_load_1,
            deformations="moment")

# 7. Define system
system = System([bar_1, bar_2, bar_3, bar_4, bar_5])

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 8. Create ReductionTheorem instance
red = ReductionTheorem(system)

print("Degree of static indeterminacy:", red.degree_of_static_indeterminacy)

# 9. Remove redundant degrees of freedom to form a statically determinate
# system
red.modify_node(node_a, 'phi')
red.modify_bar(bar_3, 'hinge_phi_j')
red.modify_bar(bar_4, 'hinge_phi_j')
red.modify_bar(bar_1, 'hinge_phi_j')
red.modify_bar(bar_2, 'hinge_phi_i')
red.modify_node(node_b, 'phi')
red.modify_node(node_f, 'u')

print("Degree of static indeterminacy (after):",
      red.degree_of_static_indeterminacy)

# Show the released system
red.plot_released_system(mode='plotly')

red.add_virtual_bar_load(bar_5, 'fz', position=0.6)

# Show virtual system
ObjectRenderer(SystemGeo(red.virtual_system), 'plotly').show()

# 10. Compute deformation directly at bar_5
delta_dv_sstatics = red.deformation()
print("Deformation (ReductionTheorem):", delta_dv_sstatics)

# 11. Hand-calculated reference value using standard formulas
red.plot('real', kind='moment', mode='plotly')
red.plot('virt', kind='moment', mode='plotly')
delta_dv_seminar = (1/4 * 2.4 * (-288) * 10 / 207900 +
                    5/12 * 2.4 * 350 * 10 / 207900)

print("Deformation (Hand calculation):", delta_dv_seminar)

# 12. Verify equality
print("Results match:",
      np.allclose(delta_dv_sstatics, delta_dv_seminar, atol=1e-3))
