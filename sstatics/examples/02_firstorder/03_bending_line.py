"""
Example 03:
Bending line of a multi-bar frame using the first-order method

This example computes and plots the bending line (deflection curve) of a
complex multi-bar frame system subjected to point and line loads. The system
contains fixed supports and various load cases.

The FirstOrder solver is used to determine the displacement field, and the
resulting bending line is visualized using the `plot(kind='bending_line')`
method.
"""

from sstatics.core.preprocessing import (
    Material, CrossSection, NodePointLoad, Node, Bar, BarLineLoad, System
)
from sstatics.core.calc_methods import FirstOrder


# 1. Define material and cross-section
mat = Material(young_mod=13700000, poisson=0.1, shear_mod=0.1,
               therm_exp_coeff=0.1)

cs = CrossSection(mom_of_int=58956e-8, area=612e-4, height=612e-4, width=1,
                  shear_cor=0.1)

# 2. Define nodes with supports and point loads
node_1 = Node(x=0, z=4, u='fixed', w='fixed', phi='fixed')
node_2 = Node(x=0, z=0)
node_3 = Node(x=3, z=0, loads=NodePointLoad(z=20))
node_4 = Node(x=11, z=0, loads=NodePointLoad(x=-15))
node_5 = Node(x=11, z=6, u='fixed', w='fixed')

# 3. Define line loads on bars
line_load_1 = BarLineLoad(pi=3.3, pj=2.4, direction='z')
line_load_2 = BarLineLoad(pi=2.4, pj=0, direction='z')
line_load_3 = BarLineLoad(pi=2, pj=2, direction='x')

# 4. Define bars and system
bar_1 = Bar(node_1, node_3, cs, mat)
bar_2 = Bar(node_2, node_3, cs, mat, line_loads=line_load_1)
bar_3 = Bar(node_3, node_4, cs, mat, line_loads=line_load_2)
bar_4 = Bar(node_4, node_5, cs, mat, line_loads=line_load_3)

system_2 = System([bar_1, bar_2, bar_3, bar_4])

# 5. Solve using first-order (linear) method
solution = FirstOrder(system_2)

# 6. Plot bending line (deflection curve)
solution.plot(kind='bending_line')

print("=== Bending Line Example ===")
print("Bending line computed and plotted for multi-bar frame system.")
