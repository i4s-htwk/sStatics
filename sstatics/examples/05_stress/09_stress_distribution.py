"""
Example 09:
Stress distribution in a beam under constant line load

This example demonstrates how to compute normal, shear, and bending stresses
along a beam using the `BarStressDistribution` class.

A simply supported beam with one fixed and one vertically restrained node
is loaded by a constant line load. After solving the system using a
first-order (linear-elastic) analysis, the stress distribution is evaluated
at 10 points along the bar.

The example prints:
- normal stresses σ(x)
- shear stresses τ(x)
- bending stresses σ_b,top(x) and σ_b,bottom(x)

Additionally, stresses are evaluated at a specific cross-section height z.
"""

from sstatics.core.preprocessing import (
    Bar, System, Material, Node, NodePointLoad, CrossSection, BarLineLoad
)
from sstatics.core.preprocessing.geometry import Polygon
from sstatics.core.calc_methods import FirstOrder


# 1. Define Material
mat = Material(210_000_000, 81_000_000, 0.1, 0.1)

# 2. Define Cross-section (rectangle 10 × 40)
rect = Polygon(points=[(0, 0), (10, 0), (10, 40), (0, 40), (0, 0)])
cs = CrossSection(geometry=[rect])

# 3. Define System: single-span beam (4 m)
n1 = Node(0, 0, u='fixed', w='fixed')
n2 = Node(4, 0, w='fixed', loads=(NodePointLoad(x=1)))

line_load = BarLineLoad(1, 1)
b1 = Bar(n1, n2, cs, mat, line_loads=line_load)
system = System([b1])

# 4. Solve system
solution = FirstOrder(system)

# 5. Stress distribution along the bar
n_disc = 10
stress_list = solution.stress_distribution(n_disc=n_disc)

for i, bar_stress in enumerate(stress_list):
    sigma = bar_stress.stress_disc
    print(f"\nBar {i}: Stress distribution at {n_disc+1} points")
    print("  σ_normal(x):        ", sigma[:, 0])
    print("  τ_shear(x):         ", sigma[:, 1])
    print("  σ_bending_bottom(x):", sigma[:, 2])
    print("  σ_bending_top(x):   ", sigma[:, 3])


# 6. Stress at a specific height z of the cross-section
z = 0

print(f"\nStress evaluation at z = {z}:")

for i, bar_stress in enumerate(stress_list):
    sigma_z = bar_stress.stress_at_z(z)
    print(f"\nBar {i}: Stress distribution at {n_disc+1} points at height "
          f"z = {z}")
    print("  σ_normal(x): ", sigma_z[:, 0])
    print("  τ_shear(x):  ", sigma_z[:, 1])
    print("  σ_bending(x):", sigma_z[:, 2])

solution.plot_stress('normal')
