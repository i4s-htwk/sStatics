"""
Example 05:
Circular full cross-section (solid circle)

This example demonstrates how to define a full circular cross-section using
the CircularSector geometry object with a 360° opening angle. The resulting
cross-section is visualized, and its geometric properties (area, centroid,
moment of inertia, boundaries) are extracted.
"""

import numpy as np
from sstatics.core.preprocessing.geometry.objects import CircularSector
from sstatics.core.preprocessing import CrossSection
from sstatics.core.postprocessing.graphic_objects import (
    CrossSectionGeo, ObjectRenderer)


# 1. Define circular sector geometry (full circle)
sector = CircularSector(
    center=(0, 0),
    radius=1.0,
    angle=np.pi * 2,        # opening angle = 360°
    start_angle=0,
    positive=True
)

# 2. Create cross-section
cs = CrossSection(geometry=[sector])

# 3. Visualize cross-section
ObjectRenderer(CrossSectionGeo(cs), 'plotly').show()

# 4. Extract geometric properties
A = cs.area
y_s = cs.center_of_mass_y
z_s = cs.center_of_mass_z
Iy = cs.mom_of_int
width = cs.width
height = cs.height

yb, zb = cs.boundary()
y_bottom, y_top = yb[0], yb[1]
z_top, z_bottom = zb[0], zb[1]

print("=== Circular Sector Cross-Section Properties ===")
print(f"Area A                     : {A:.6f}")
print(f"Centroid y_s, z_s         : ({y_s:.6f}, {z_s:.6f})")
print(f"Moment of inertia Iy      : {Iy:.6f}")
print(f"Width, Height             : {width:.6f}, {height:.6f}")
print(f"Boundary in y-direction   : bottom = {y_bottom:.6f},"
      f" top = {y_top:.6f}")
print(f"Boundary in z-direction   : top = {z_top:.6f},"
      f" bottom = {z_bottom:.6f}")
