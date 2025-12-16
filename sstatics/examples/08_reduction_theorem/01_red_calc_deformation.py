"""
Example 01:
Computing the vertical deformation w at midspan of Bar 2
using the Reduction Theorem

In this example, the vertical deformation w at the midpoint of Bar 2 is
computed using the Reduction Theorem. The frame is initially statically
indeterminate with a degree of indeterminacy equal to 1. By introducing a
hinge at the right end of Bar 2, the system becomes statically determinate
and suitable for applying the Reduction Theorem.

To obtain the desired deformation at midspan, a virtual vertical load is
applied at position x = 0.5 of Bar 2. The deformation is computed from the
work equation of the virtual system.

To verify the correctness of the procedure, the same deformation is
calculated analytically using classical integration-table formulas. Both
results agree within numerical tolerance, confirming the correctness of the
computed internal forces and the Reduction Theorem implementation.
"""

# 1. Import required modules
from sstatics.core.calc_methods import ReductionTheorem
from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System, BarLineLoad
)
import numpy as np

# 2. Define material and cross-section
material = Material(21000, 1, 1, 1)
cs = CrossSection(1, 1, 1, 1, 1)

# 3. Define frame nodes
n1 = Node(0, 0, u='fixed', w='fixed')
n2 = Node(0, -2)
n3 = Node(4, -2)
n4 = Node(4, 0, u='fixed', w='fixed')

# 4. Define loads
line_load = BarLineLoad(pi=2.5, pj=2.5, direction='z')

# 5. Define bars
bar_1 = Bar(n1, n2, cs, material, deformations="moment")
bar_2 = Bar(n2, n3, cs, material, line_loads=line_load,
            deformations="moment")
bar_3 = Bar(n3, n4, cs, material, deformations="moment")

system = System([bar_1, bar_2, bar_3])

# 6. Initialize Reduction Theorem
red = ReductionTheorem(system)

print("Degree of static indeterminacy (before):",
      red.degree_of_static_indeterminacy)
# → 1 (one redundant constraint must be removed)

# 7. Remove the redundant rotational constraint by inserting a hinge
red.modify_bar(bar_2, 'hinge_phi_j')
print("Degree of static indeterminacy (after):",
      red.degree_of_static_indeterminacy)
# → 0 (system is now statically determinate)

# Show the released system
red.plot_released_system()

# 8. Apply a virtual unit vertical load at midspan of Bar 2
red.add_virtual_bar_load(bar_2, 'fz', position=0.5)

# 9. Compute deformation using the Reduction Theorem
delta_reduction = red.deformation()
print("w(midspan) (Reduction Theorem):", delta_reduction)
# Expected: ~1.587747932183352e-04 m

# 10. Hand calculation using classical integration tables
red.plot('real', kind='moment')
red.plot('virt', kind='moment')

delta_hand = (
        1/2 * 1 * (-2.5) * 4 / 21000 +
        5/12 * 1 * 5 * 4 / 21000
)
print("w(midspan) (Hand calculation):", delta_hand)
# Expected: ~1.5873015873e-04 m

# 11. Verify equality of both methods
print("Results match:",
      np.allclose(delta_reduction, delta_hand, atol=1e-7))
