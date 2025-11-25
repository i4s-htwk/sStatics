"""
Example 09:
Stress distribution in a bar under a constant line load

This example demonstrates how to compute normal, shear, and bending stresses
along a beam using the `BarStressDistribution` class.

A single-span beam (Einfeldträger) is defined between two nodes:
- Node 1: fixed in horizontal (u) and vertical (w) displacement
- Node 2: fixed only in vertical displacement

A constant line load is applied to the bar. The system is solved with a
first-order (linear-elastic) analysis using the displacement method.
The resulting deformation and internal forces are then used to compute
stress distributions.

The stress distribution is evaluated at 10 equally spaced points along the bar.
Finally, the normal, shear, and bending stresses are printed and plotted.
"""

from sstatics.core.preprocessing import (
    Bar, System, Material, Node, CrossSection, BarLineLoad
)
from sstatics.core.preprocessing.geometry import Polygon
from sstatics.core.calc_methods import FirstOrder
from sstatics.core.postprocessing import BarStressDistribution


# 1. Define material
mat = Material(210_000_000, 0.1, 81_000_000, 0.1)  # steel S235


# 2. Define rectangular cross-section (width=10, height=40)
rect = Polygon(points=[(0, 0), (10, 0), (10, 40), (0, 40), (0, 0)])
cs = CrossSection(geometry=[rect])


# 3. Create a simple single-span beam (Einfeldträger)
n1 = Node(0, 0, u='fixed', w='fixed')
n2 = Node(4, 0, w='fixed')

# Constant line load along the bar
line_load = BarLineLoad(1, 1)

# Create bar and system
b1 = Bar(n1, n2, cs, mat, line_loads=line_load)
system = System([b1])


# 4. Solve with first-order theory (linear elastic)
solution = FirstOrder(system)


# 5. Compute stress distributions from solution
stressbar = BarStressDistribution(
    bar=b1,
    deform=solution.bar_deform_list[0],
    force=solution.internal_forces[0],
    disc=10  # number of sampling points
)

# Stress types
normal_stresses = stressbar.compute('normal')
shear_stresses = stressbar.compute('shear', z=0.5)
bending_stresses = stressbar.compute('bending')


# 6. Output results
print("Normal stresses (array at discretization points):")
print(normal_stresses)

print("Shear stresses at z = 0.5:")
print(shear_stresses)

print("Bending stresses:")
print(bending_stresses)


# 7. Plot stress distributions
stressbar.plot('normal')
stressbar.plot('bending')
stressbar.plot('shear')
