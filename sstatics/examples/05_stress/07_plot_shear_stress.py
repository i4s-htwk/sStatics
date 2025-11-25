"""
Example 07:
Plotting shear stress
"""

from sstatics.core.preprocessing import CrossSection
from sstatics.core.preprocessing.geometry.objects import Polygon
from sstatics.core.postprocessing import CrossSectionStress

# 1. Create a simple rectangular cross-section (width 10, height 40)
rect = Polygon(points=[(0, 0), (10, 0), (10, 40), (0, 40), (0, 0)])
cs = CrossSection(geometry=[rect])

# 2. Stress calculator
stress = CrossSectionStress(cs)

# 3. Example loads
V = 2000

# shear_stress_disc requires bottom (z_i) and top (z_j) coordinates.
# For this rectangle: z = 0 at the bottom, z = 10 at the top.
#
# n_disc = 20 means:
# → 20 intermediate points are created between z_i and z_j
# → resulting in 21 evaluation points including both boundaries.
#
# At each of these z-positions, the shear stress τ(z) is computed
# and stored internally for later plotting.
stress.shear_stress_disc(v_z=V, z_i=0, z_j=10, n_disc=20)

# 4. Plot stored distribution
stress.plot(kind="shear")
