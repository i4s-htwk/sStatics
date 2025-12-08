"""
Example 08:
Stress in a T-shaped cross-section
"""

from sstatics.core.preprocessing import CrossSection
from sstatics.core.preprocessing.geometry.objects import Polygon
from sstatics.core.postprocessing import CrossSectionStress


# 1. Define T-shaped geometry (flange + web)
geometry = [
    Polygon([(0, 0), (30, 0), (30, 3), (0, 3), (0, 0)]),      # flange
    Polygon([(14, 3), (16, 3), (16, 43), (14, 43), (14, 3)])  # web
]

cs = CrossSection(geometry=geometry)

# 2. Stress calculator
stress = CrossSectionStress(cs)

# 3. Example loads
N = 2   # axial force
M = 10  # bending moment
V = 10  # shear force

# Centroid location
z_s = cs.center_of_mass_z

# Cross-section boundaries
_, zb = cs.boundary()
z_top, z_bottom = zb[0], zb[1]

# Distances to centroid
dist_top = abs(z_top - z_s)
dist_bottom = abs(z_bottom - z_s)

print("=== T Cross-Section Example ===")
print("Centroid at z_s =", z_s)
print(f"Distance top to centroid: {dist_top}")
print(f"Distance bottom to centroid: {dist_bottom}")

# Normal stress: maximum occurs everywhere
print("Normal stress (maximum):", stress.normal_stress(N))

# Bending stress: maximum occurs at largest distance from centroid
# Distances: top = {dist_top}, bottom = {dist_bottom}
# z = z_bottom (43) has the larger distance → maximum bending stress
# If z is not specified, the method automatically returns the maximum
# The distance calculation is just for illustration of how the method works

# Automatically returns maximum bending stress
print("Bending stress (maximum):", stress.bending_stress(M))

# Explicitly at the largest distance from centroid (illustrative)
print("Bending stress at the largest distance from centroid:",
      stress.bending_stress(M, z=z_bottom))

# Shear stress: maximum occurs at centroid
print("Shear stress (maximum at centroid):", stress.shear_stress(V))

# Shear distribution for plotting along the web
stress.shear_stress_disc(v_z=V, z_i=3.01, z_j=43, n_disc=20)
stress.plot(kind="shear")
