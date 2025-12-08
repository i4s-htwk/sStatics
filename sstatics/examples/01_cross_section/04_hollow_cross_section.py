"""
Example 04:
Hollow square profile with rounded corners

DIN EN 10210-2

This example demonstrates how to create a hollow square cross-section using
Polygons and CircularSectors for outer and inner fillets. The cross-section is
visualized, and its geometric properties (area, centroid, moment of inertia,
boundaries) are extracted.
"""

import numpy as np
from sstatics.core.preprocessing.geometry.objects import (
    Polygon, CircularSector)
from sstatics.core.preprocessing import CrossSection
from sstatics.graphic_objects import CrossSectionGraphic


# Parameters
outer_dim = 40
t = 5
inner_dim = outer_dim - 2 * t

r_inner = 5      # inner fillet radius
r_outer = 7.5    # outer fillet radius

# Half dimensions
h = outer_dim / 2
h_i = inner_dim / 2

# 1. Outer rectangle (without rounded corners)
outer_rect = Polygon([
    (-h + r_outer, -h), (h - r_outer, -h),
    (h, -h + r_outer), (h, h - r_outer),
    (h - r_outer,  h), (-h + r_outer,  h),
    (-h,  h - r_outer), (-h, -h + r_outer),
    (-h + r_outer, -h)
])

# 2. Inner rectangle (cut-out)
inner_rect = Polygon([
    (-h_i + r_inner, -h_i), (h_i - r_inner, -h_i),
    (h_i, -h_i + r_inner), (h_i,  h_i - r_inner),
    (h_i - r_inner,  h_i), (-h_i + r_inner,  h_i),
    (-h_i,  h_i - r_inner), (-h_i, -h_i + r_inner),
    (-h_i + r_inner, -h_i)
], positive=False)

# 3. Outer fillets (4 quarter circles)
outer_arcs = [
    CircularSector(center=(h - r_outer, h - r_outer), radius=r_outer,
                   angle=np.pi/2, start_angle=0, positive=True),
    CircularSector(center=(-h + r_outer, h - r_outer), radius=r_outer,
                   angle=np.pi/2, start_angle=-3*np.pi/2, positive=True),
    CircularSector(center=(-h + r_outer, -h + r_outer), radius=r_outer,
                   angle=np.pi/2, start_angle=np.pi, positive=True),
    CircularSector(center=(h - r_outer, -h + r_outer), radius=r_outer,
                   angle=np.pi/2, start_angle=-np.pi/2, positive=True)
]

# 4. Inner fillets (negative quarter circles)
inner_arcs = [
    CircularSector(center=(h_i - r_inner,  h_i - r_inner), radius=r_inner,
                   angle=np.pi/2, start_angle=0, positive=False),
    CircularSector(center=(-h_i + r_inner,  h_i - r_inner), radius=r_inner,
                   angle=np.pi/2, start_angle=-3*np.pi/2, positive=False),
    CircularSector(center=(-h_i + r_inner, -h_i + r_inner), radius=r_inner,
                   angle=np.pi/2, start_angle=-np.pi, positive=False),
    CircularSector(center=(h_i - r_inner, -h_i + r_inner), radius=r_inner,
                   angle=np.pi/2, start_angle=-np.pi/2, positive=False)
]

# 5. Construct final cross-section
geometry = [outer_rect] + outer_arcs + [inner_rect] + inner_arcs
cs = CrossSection(geometry=geometry)

# 6. Visualize cross-section
CrossSectionGraphic(cross_section=cs, merged=False).show()
CrossSectionGraphic(cross_section=cs).show()

# 7. Extract geometric properties
A = cs.area
y_s = cs.center_of_mass_y
z_s = cs.center_of_mass_z
Iy = cs.mom_of_int
width = cs.width
height = cs.height

yb, zb = cs.boundary()
y_bottom, y_top = yb[0], yb[1]
z_top, z_bottom = zb[0], zb[1]

print("=== Hollow Square Cross-Section Properties ===")
print(f"Area A                     : {A:.6f}")
print(f"Centroid y_s, z_s         : ({y_s:.6f}, {z_s:.6f})")
print(f"Moment of inertia Iy      : {Iy:.6f}")
print(f"Width, Height             : {width:.6f}, {height:.6f}")
print(f"Boundary in y-direction   : bottom = {y_bottom:.6f},"
      f" top = {y_top:.6f}")
print(f"Boundary in z-direction   : top = {z_top:.6f},"
      f" bottom = {z_bottom:.6f}")
