"""
Example 01:
Demonstration of the Force Method for a simple frame

In this example, a small frame with three bars is analyzed using the
Force Method. The method introduces redundant forces corresponding to
the removed constraints (hinges) and computes the internal force
distribution using influence coefficients (delta_i_j) and load coefficients
(delta_i_0).

The system includes:
- Line and point loads on the bars
- Hinges introduced at specific bar ends
- Computation of redundant forces by solving A * x = b
"""

# 1. Import required modules
from sstatics.core.calc_methods import ForceMethod
from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System, BarLineLoad, BarPointLoad
)
import numpy as np

# 2. Define material and cross-sections
mat_1 = Material(3.0e7, 1, 1, 0.000012)
cs_1 = CrossSection(0.003125, 0.15, 0.5, 0.3, 1.0)
cs_2 = CrossSection(0.000675, 0.09, 0.3, 0.3, 1.0)

# 3. Define nodes
n1 = Node(0, 0, u="free", w="fixed", phi="free")
n2 = Node(2, 0)
n3 = Node(5, 0, u="free", w="fixed", phi="free")
n4 = Node(2, 2.5, u="fixed", w="fixed", phi="fixed")

# 4. Define loads
line_load = BarLineLoad(pi=30, pj=30, direction="z")
point_load = BarPointLoad(x=60, position=0.6)
def_comp = 'moment'

# 5. Define bars
b1 = Bar(n1, n2, cs_1, mat_1, line_loads=line_load, deformations=def_comp)
b2 = Bar(n2, n3, cs_1, mat_1, deformations=def_comp)
b3 = Bar(n4, n2, cs_2, mat_1, point_loads=point_load, deformations=def_comp)

bars = [b1, b2, b3]

# 6. Create system
system = System(bars)

# 7. Initialize Force Method
force_method = ForceMethod(system)

print("Degree of static indeterminacy:",
      force_method.degree_of_static_indeterminacy)

# 8. Remove redundant degrees of freedom to form a statically determinate
# system
force_method.modify_bar(b1, hinge="hinge_phi_j")
force_method.modify_bar(b3, hinge="hinge_phi_j")

print("Degree of static indeterminacy:",
      force_method.degree_of_static_indeterminacy)

# plot released system
# force_method.plot_released_system()

# 9. Compute influence coefficients (delta_i_j = Vorzahlen)
#    and load coefficients (delta_i_0 = Belastungszahlen)
delta_i_j = np.array(force_method.influence_coef)
delta_i_0 = np.array(force_method.load_coef)

# 10. Display system of equations A * x = b
print("System of equations (delta_i_j * x = delta_i_0):")
for i in range(delta_i_j.shape[0]):
    row = " + ".join(f"{delta_i_j[i,j]:.4e}*x{j+1}"
                     for j in range(delta_i_j.shape[1]))
    print(f"{row} = {delta_i_0[i,0]:.4e}")

# 11. Solve for redundant forces (x = unbekannte Kräfte)
x = np.array(force_method.redundants)

print("Redundant forces (x):")
for i in range(x.shape[0]):
    print(f"x{i+1} = {x[i, 0]:.6e}")

# 11. Optional visualizations
force_method.plot(mode='rls', kind='moment')    # released system
force_method.plot(mode='uls', uls_index=0, kind='moment')  # ULS 1
force_method.plot(mode='uls', uls_index=1, kind='moment')  # ULS 2

# 12. Inspecting the work matrix:
#
# To understand the origin of delta_i_j and delta_i_0 more deeply,
# we can inspect the internal work contributions of individual bars.
#
# Example:
# - uls_index_i = 0:   ULS 1
# - uls_index_j = 1:   ULS 2

print("\nWork matrix for ULS1 × ULS2:")
print(force_method.work_matrix('bars', uls_index_i=0, uls_index_j=1))

print("\nWork contributions of bar b3 for ULS1 × ULS2:")
print(force_method.work_of(
    obj=b3,
    uls_index_i=0,
    uls_index_j=1,
    sum=False
))
