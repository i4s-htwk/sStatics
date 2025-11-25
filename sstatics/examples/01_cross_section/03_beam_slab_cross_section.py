"""
Example 03:
Composite cross-section consisting of a concrete slab and a supporting beam.

This example demonstrates how separate geometric parts (slab + beam)
are combined into one unified cross-section. The resulting geometric
properties such as area, centroid, moment of inertia, and boundaries
are extracted afterwards.
"""

from sstatics.core.preprocessing.geometry import Polygon
from sstatics.core.preprocessing import CrossSection
from sstatics.graphic_objects import CrossSectionGraphic


# 1. Define geometry
# Slab: width = 200, thickness = 24
slab = Polygon([
    (0, 0), (200, 0), (200, 24), (0, 24), (0, 0)
])

# Beam: width = 30, height = 60, centered under the slab
beam = Polygon([
    (85, 24), (115, 24), (115, 84), (85, 84), (85, 24)
])

# 2. Create cross-section
beam_with_slab = CrossSection(geometry=[slab, beam])

# 3. Visualize geometry objects (before merging)
CrossSectionGraphic(cross_section=beam_with_slab, merged=False).show()

# 4. Visualize merged cross-section
CrossSectionGraphic(cross_section=beam_with_slab).show()

# 5. Extract geometric properties
A = beam_with_slab.area
y_s = beam_with_slab.center_of_mass_y
z_s = beam_with_slab.center_of_mass_z
Iy = beam_with_slab.mom_of_int
width = beam_with_slab.width
height = beam_with_slab.height

(yb, zb) = beam_with_slab.boundary()
y_bottom, y_top = yb[0], yb[1]
z_top, z_bottom = zb[0], zb[1]

# 6. Print results
print("=== Composite Cross-Section: Beam with Slab ===")
print(f"Area A                     : {A:.4f}")
print(f"Centroid y_s, z_s         : ({y_s:.4f}, {z_s:.4f})")
print(f"Moment of inertia Iy      : {Iy:.4f}")
print(f"Width, Height             : {width:.4f}, {height:.4f}")
print(f"Boundary in y-direction   : bottom = {y_bottom:.4f},"
      f" top = {y_top:.4f}")
print(f"Boundary in z-direction   : top = {z_top:.4f},"
      f" bottom = {z_bottom:.4f}")
