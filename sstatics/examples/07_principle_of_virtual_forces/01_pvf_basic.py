"""
Example 01:
Computing the displacement w at a node using the
Principle of Virtual Forces (PVF, german:Prinzip der virtuellen Kräfte)

In this example, the vertical displacement w at node_2 of a cantilever beam
is computed using two independent methods:

1. The displacement method (FirstOrder)
2. The Principle of Virtual Forces (PVF)

The principle of virtual forces provides a powerful and elegant way to
determine displacements by introducing a virtual load in the direction and at
the location of the desired displacement. Evaluating the work equation yields
the desired deformation.

To validate the PVF result, the same displacement w at node_2 is computed
using the FirstOrder solver. The results of both methods are compared and
verified to be equal within numerical tolerance.
"""

# 1. Import required modules
from sstatics.core.preprocessing import (
    Bar, CrossSection, Material, Node, NodePointLoad, System
)
from sstatics.core.preprocessing.geometry import Polygon
from sstatics.core.calc_methods import FirstOrder, PVF
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)
import numpy as np

# 2. Define cross-section using polygon geometry
cs_1 = CrossSection(
    geometry=[Polygon([(0, 0), (0.1, 0), (0.1, 0.2), (0, 0.2), (0, 0)])]
)

# 3. Define material (E = 11,000,000 kN/m²)
mat_1 = Material(11000000, 0.1, 0.1, 0.1)

# 4. Define nodes
n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')  # Fixed support
n2 = Node(3, 0, loads=NodePointLoad(z=1))  # Real vertical load at free end

# 5. Define bar and system
bar = Bar(n1, n2, cs_1, mat_1)
system = System([bar])

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 6. Compute displacement using Principle of Virtual Forces (PVK)
pvf = PVF(system)

# For displacement w at node_2:
# → Apply a virtual vertical force fz at the bar end (position=1)
pvf.add_virtual_bar_load(obj=bar, force='fz', position=1)

# Show virtual system
ObjectRenderer(SystemGeo(pvf.virtual_system), 'plotly').show()

w_node_2_pvf = pvf.deformation()
print("w at node 2 (PVK) [m]:", w_node_2_pvf)

# 7. Compute displacement using FirstOrder method
solution = FirstOrder(system)

# Extract deformation vector of the single bar (index 0)
bar_deform = solution.bar_deform_total[0]

# Vertical displacement w at node_2 is located at index 4
w_node_2 = bar_deform[4][0]
print("w at node 2 (FirstOrder) [m]:", w_node_2)

# 8. Verify equality of both methods
print("Results match:", np.allclose(w_node_2, w_node_2_pvf))

# 9. Plot moment distribution real loads
pvf.plot('real', kind='moment')

# 9. Plot moment distribution virtual loads
pvf.plot('virt', kind='moment')
