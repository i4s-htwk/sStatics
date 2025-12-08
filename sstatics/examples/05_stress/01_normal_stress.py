"""
Example 01:
Basic normal stress calculation σ = N / A
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

# 4. Calculate stress
sigma = stress.normal_stress(n=N)

print("=== Normal Stress Example ===")
print(f"Cross-section area A     = {cs.area:.3f}")
print(f"Applied normal force N    = {N}")
print(f"Normal stress sigma = N/A = {sigma:.6f}")
