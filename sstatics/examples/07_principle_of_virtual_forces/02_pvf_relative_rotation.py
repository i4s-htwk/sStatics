"""
Example 02:
Computing the relative rotation of two bars at a shared node using
the Principle of Virtual Forces (PVF)

In this example, the relative rotation (phi) of bars 1 and 2 at the
common node_3 is computed. Individual deformation contributions
can be specified for each bar using the `deformations` attribute.

The PVF method introduces a virtual moment couple on the two bars at
the connecting node. Evaluating the work equation yields the desired
relative rotation.
"""

# 1. Import modules
from sstatics.core.preprocessing import (
    Bar, BarLineLoad, CrossSection, Material, Node, NodePointLoad, System
)
from sstatics.core.calc_methods import PVF
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)
import numpy as np

# 2. Define material
s253 = Material(210000000, 0.1, 81000000, 0.1)

# 3. Define cross-sections
ipe_360 = CrossSection(16270 * 1e-8, 72.7 * 1e-4, 0.360, 0.170, 5/6)
ipe_220 = CrossSection(2772 * 1e-8, 33.4 * 1e-4, 0.240, 0.120, 5/6)

# 4. Define nodes
n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed', rotation=np.pi/2)
n2 = Node(0, -3)
n3 = Node(3, -3)
n4 = Node(7, -3, loads=[NodePointLoad(x=-20)])
n5 = Node(9, 0, u='fixed', w='fixed')

# 5. Define loads and deformation components
line_load = BarLineLoad(pi=12, pj=12)
def_comp = 'moment'

# 6. Define bars
b1 = Bar(n1, n2, ipe_360, s253, deformations=def_comp)
b2 = Bar(n2, n3, ipe_360, s253, line_loads=line_load, deformations=def_comp)
b3 = Bar(n3, n4, ipe_220, s253, hinge_phi_i=True, hinge_phi_j=True,
         line_loads=line_load, deformations=def_comp)

b4 = Bar(n4, n5, ipe_220, s253, deformations=def_comp)

# 7. Assemble system
system = System(bars=[b1, b2, b3, b4])

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 8. Compute relative rotation using PVF
pvf = PVF(system)

# Apply a virtual moment couple on bars 2 and 3 at node 3
pvf.add_virtual_moment_couple(
    bar_positive_m=b2,
    bar_negative_m=b3,
    connecting_node=n3
)

# Show virtual system
ObjectRenderer(SystemGeo(pvf.virtual_system), 'plotly').show()

# Plot moment of the virtual and the real system
pvf.plot(mode='virt', kind='moment')
pvf.plot(mode='real', kind='moment')

delta_phi = pvf.deformation()
print("Relative rotation at node 3 (PVK) [rad]:", delta_phi)

# The total deformation results from evaluating the work contributions
# of both bars and nodes according to the principle of virtual forces.
# Using the work matrices, we can inspect how each bar and each node
# contributes to the overall work equation individually.
# Each column in the matrices corresponds to a specific bar or node,
# allowing detailed insight into the distribution of internal forces
# and resulting deformations.

# Work matrix for bars
work_matrix_bars = pvf.work_matrix(kind='bars')
print("Work matrix (bars):\n", work_matrix_bars)

# Work matrix for nodes
work_matrix_nodes = pvf.work_matrix(kind='nodes')
print("Work matrix (nodes):\n", work_matrix_nodes)

# If we want to query the specific contribution of a single model object
# to the work equation, we can use the `work_of(...)` method.
work_b4 = pvf.work_of(obj=b1)
print("Work contribution of bar 1:\n", work_b4)
