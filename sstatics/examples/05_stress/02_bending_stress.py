"""
Example 02:
Bending stress σ = M * z / Iy
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
M = 50000

# 4. Boundaries
_, zb = cs.boundary()
z_top, z_bottom = zb[0], zb[1]

# 5. Calculate Stress
sigma_bottom = stress.bending_stress(m_yy=M, z=zb[0])
sigma_top = stress.bending_stress(m_yy=M, z=zb[1])

print("=== Bending Stress Example ===")
print(f"Moment of inertia Iy = {cs.mom_of_int:.3f}")
print(f"Bending moment M     = {M}")
print(f"σ_bottom             = {sigma_bottom:.6f}")
print(f"σ_top                = {sigma_top:.6f}")
