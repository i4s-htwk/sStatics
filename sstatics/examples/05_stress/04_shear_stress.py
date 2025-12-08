"""
Example 04:
Shear stress τ(z) using Jourawski’s formula
    τ(z) = V * S(z) / (I * t(z))

Note:
- This method assumes thin-walled cross-sections.
- For thick rectangles or solid sections, τ(z) is not constant across
  thickness, so results may be inaccurate.
"""
import numpy as np
from sstatics.core.preprocessing import CrossSection
from sstatics.core.preprocessing.geometry.objects import Polygon
from sstatics.core.postprocessing import CrossSectionStress


# 1. Create a simple rectangular cross-section (width 10, height 40)
rect = Polygon(points=[(0, 0), (10, 0), (10, 40), (0, 40), (0, 0)])
cs = CrossSection(geometry=[rect])

# 2. Create stress calculator
stress = CrossSectionStress(cs)

# 3. Example loads
V = 2000

# 4. Boundaries
_, zb = cs.boundary()
z_top, z_bottom = zb[0], zb[1]

print("=== Shear Stress Example ===")
print(f"\nTop boundary: z = {z_top}, Bottom boundary: z = {z_bottom}")

# 5. Maximum shear stress occurs at the centroid (default)
tau_max = stress.shear_stress(v_z=V)
print("Maximum shear stress (at centroid):", tau_max)

# 6. Shear stress distribution along height
z_values = np.linspace(z_top, z_bottom, 9)
print("\nShear stress distribution along height:")
for z in z_values:
    tau = stress.shear_stress(v_z=V, z=z)
    print(f"z = {z:6.2f}  tau = {tau:10.6f}")
