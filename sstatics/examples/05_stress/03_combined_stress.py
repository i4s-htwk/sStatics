"""
Example 03:
Combined stress σ = N/A + M*z/I
"""

from sstatics.core.preprocessing import CrossSection
from sstatics.core.preprocessing.geometry.objects import Polygon
from sstatics.core.postprocessing import CrossSectionStress


# 1. Create a simple rectangular cross-section (width 10, height 40)
rect = Polygon(points=[(0, 0), (10, 0), (10, 40), (0, 40), (0, 0)])
cs = CrossSection(geometry=[rect])

# 2. Create stress calculator
stress = CrossSectionStress(cs)

# 3. Example loads
N = 1000
M = 50000

z = cs.center_of_mass_z + 5  # 5 mm above centroid

# 4. Calculate Stress
sigma_combined = stress.combine_axial_bending_stress(n=N, m_yy=M, z=z)
z0 = stress.zero_line(n=N, m_yy=M)

print("=== Combined Stress Example ===")
print(f"N = {N}, M = {M}, z = {z}")
print(f"σ_combined = {sigma_combined:.6f}")
print(f"Zero-stress line at z0 = {z0:.3f}")
