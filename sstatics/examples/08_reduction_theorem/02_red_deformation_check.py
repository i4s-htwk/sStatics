"""
Example 02:
Checking the deformation condition of a statically indeterminate frame
using the Reduction Theorem

In this example, a single-degree statically indeterminate frame is analyzed
using the Reduction Theorem. The method removes one redundant constraint,
forms a statically determinate system, and evaluates the deformation
condition associated with the removed constraint.

A hinge is introduced at the right end of the middle bar, reducing the
degree of static indeterminacy from 1 to 0. The deformation at this location
must be zero in the real indeterminate system. By applying an appropriate
virtual moment couple at the released joint, the Reduction Theorem computes
this deformation.

To validate the implementation, a hand-calculated reference value based on
standard integration tables is evaluated. Both results are very close to
zero, confirming the correctness of the internal moment distribution and the
Reduction Theorem procedure.
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

# 3. Define nodes (fixed–free–free–fixed frame geometry)
n1 = Node(0, 0, u='fixed', w='fixed')
n2 = Node(0, -2)
n3 = Node(4, -2)
n4 = Node(4, 0, u='fixed', w='fixed')

# 4. Define loads
line_load = BarLineLoad(pi=2.5, pj=2.5, direction='z')

# 5. Define bars
bar_1 = Bar(n1, n2, cs, material, deformations="moment")
bar_2 = Bar(n2, n3, cs, material, line_loads=line_load, deformations="moment")
bar_3 = Bar(n3, n4, cs, material, deformations="moment")

system = System([bar_1, bar_2, bar_3])

# 6. Degree of static indeterminacy
red = ReductionTheorem(system)
print("Degree of static indeterminacy (before):",
      red.degree_of_static_indeterminacy)
# → 1

# 7. Introduce a hinge to remove the redundant constraint
red.modify_bar(bar_2, 'hinge_phi_j')
print("Degree of static indeterminacy (after):",
      red.degree_of_static_indeterminacy)
# → 0

# 8. Apply a virtual moment couple at the released joint
#    This corresponds to the removed rotational constraint.
red.add_virtual_moment_couple(bar_2, bar_3, n3)

# 9. Compute deformation using the Reduction Theorem
delta_reduction = red.deformation()
print("δ (Reduction Theorem):", delta_reduction)

# 10. Hand-calculated reference value using standard integration tables
red.plot('real', kind='moment')
red.plot('virt', kind='moment')

delta_hand = (
        2 * 1/3 * 1 * (-2.5) * 2 / 21000
        + 2/3 * 1 * 5 * 4 / 21000
        + 1 * 1 * (-2.5) * 4 / 21000
)
print("δ (Hand calculation):", delta_hand)

# 11. Verify equality (both must be approximately zero)
print("Results match:",
      np.allclose(delta_reduction, delta_hand, atol=1e-6))
