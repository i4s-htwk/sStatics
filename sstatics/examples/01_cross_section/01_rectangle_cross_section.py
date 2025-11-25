"""
Example 01:
Simple rectangular cross-section

This example demonstrates how to create a basic rectangular cross-section
and extract its fundamental geometric properties such as area, centroid,
moments of inertia, and boundaries.
"""

from sstatics.core.preprocessing.geometry.objects import Polygon
from sstatics.core.preprocessing import CrossSection
from sstatics.graphic_objects import CrossSectionGraphic


# 1. Define rectangle (width = 40, height = 40)
rect = Polygon(
    points=[(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
)

# 2. Create cross-section
cs = CrossSection(geometry=[rect])

# 3. Visualize geometry
CrossSectionGraphic(cross_section=cs).show()

# 4. Extract properties
A = cs.area
y_s = cs.center_of_mass_y
z_s = cs.center_of_mass_z
Iy = cs.mom_of_int
width = cs.width
height = cs.height

yb, zb = cs.boundary()
y_top, y_bottom = yb[1], yb[0]
z_top, z_bottom = zb[0], zb[1]

print("=== Rectangular Cross-Section Properties ===")
print(f"Area A                     : {A:.4f}")
print(f"Centroid y_s, z_s         : ({y_s:.4f}, {z_s:.4f})")
print(f"Moment of inertia Iy      : {Iy:.3f}")
print(f"Width, Height             : {width:.4f}, {height:.4f}")
print(f"Boundary in y-direction   : top = {y_top:.4f}, "
      f"bottom = {y_bottom:.4f}")
print(f"Boundary in z-direction   : top = {z_top:.4f}, "
      f"bottom = {z_bottom:.4f}")
