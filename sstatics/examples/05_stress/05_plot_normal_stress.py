"""
Example 05:
Plotting normal stress
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

# We must call *_disc() before plotting:
# - It computes stress values at specific section heights
# - And stores them internally for plot()
disc_values = stress.normal_stress_disc(n=N)
print(disc_values)

# Structure of disc_values:
# [
#   [z-values],         # positions along height
#   [stress-values]     # sigma at each position
# ]

# 4. Plot stored distribution
stress.plot(kind="normal")
