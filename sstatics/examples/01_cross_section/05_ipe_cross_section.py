"""
Example 05:
IPE-80 profile

nach DIN 1025-5, EURONORM 53-62, DIN EN 10034

This example constructs an IPE-like I-section using:
- simple polygons for flanges and web,
- four *square filler blocks* (positive polygons) placed at the web–flange
  intersections,
- four *quarter-circle cutouts* (negative CircularSector objects) to create
  realistic rounded fillets (radius r).

This combination reproduces a rounded I-profile despite polygon geometry
constraints.

The cross-section is visualized and basic geometric properties are printed.

Parameters (units arbitrary):
- height = 80
- width  = 46
- web thickness = 3.8
- flange thickness = 5.2
- fillet radius = 5
"""

import numpy as np
from sstatics.core.preprocessing.geometry import Polygon, CircularSector
from sstatics.core.preprocessing import CrossSection
from sstatics.core.postprocessing.graphic_objects import (
    CrossSectionGeo, ObjectRenderer)


# Parameters
height = 80.0
width = 46.0
tw = 3.8    # web thickness
tf = 5.2    # flange thickness
r = 5.0    # rounding radius

# Half-dimensions, center at (0,0)
h2 = height / 2.0
b2 = width / 2.0

# 1. Base polygons: flanges and web
# Top flange
top_flange = Polygon([
    (-b2, h2), (b2, h2),
    (b2, h2 - tf), (-b2, h2 - tf),
    (-b2, h2)
])

# Bottom flange
bottom_flange = Polygon([
    (-b2, -h2 + tf), (b2, -h2 + tf),
    (b2, -h2), (-b2, -h2),
    (-b2, -h2 + tf)
])

# Web (centered)
web = Polygon([
    (-tw/2.0, -h2 + tf), (tw/2.0, -h2 + tf),
    (tw/2.0,  h2 - tf), (-tw/2.0,  h2 - tf),
    (-tw/2.0, -h2 + tf)
])

# 2. Square filler blocks (positive)
# These squares fill the corner region before rounding is cut out.
# They ensure that subtracting the circular sectors creates correct fillets.

sq_tr = Polygon([
    (tw/2.0, h2 - tf),
    (tw/2.0 + r, h2 - tf),
    (tw/2.0 + r, h2 - tf - r),
    (tw/2.0, h2 - tf - r),
    (tw/2.0, h2 - tf)
])

sq_tl = Polygon([
    (-tw/2.0, h2 - tf),
    (-tw/2.0 - r, h2 - tf),
    (-tw/2.0 - r, h2 - tf - r),
    (-tw/2.0, h2 - tf - r),
    (-tw/2.0, h2 - tf)
])

sq_bl = Polygon([
    (-tw/2.0, -h2 + tf),
    (-tw/2.0 - r, -h2 + tf),
    (-tw/2.0 - r, -h2 + tf + r),
    (-tw/2.0, -h2 + tf + r),
    (-tw/2.0, -h2 + tf)
])

sq_br = Polygon([
    (tw/2.0, -h2 + tf),
    (tw/2.0 + r, -h2 + tf),
    (tw/2.0 + r, -h2 + tf + r),
    (tw/2.0, -h2 + tf + r),
    (tw/2.0, -h2 + tf)
])

fillet_blocks = [sq_tr, sq_tl, sq_bl, sq_br]

# 3. Circular fillets (negative cutouts)
# These create the real rounding by *removing* quarter-circle regions.

ctr_tr = (tw/2.0 + r, h2 - tf - r)
ctr_tl = (-tw/2.0 - r, h2 - tf - r)
ctr_bl = (-tw/2.0 - r, -h2 + tf + r)
ctr_br = (tw/2.0 + r, -h2 + tf + r)

outer_arcs = [
    CircularSector(center=ctr_bl, radius=r, angle=np.pi/2,
                   start_angle=-np.pi/2, positive=False),
    CircularSector(center=ctr_br, radius=r, angle=np.pi/2,
                   start_angle=np.pi, positive=False),
    CircularSector(center=ctr_tr, radius=r, angle=np.pi/2,
                   start_angle=np.pi/2, positive=False),
    CircularSector(center=ctr_tl, radius=r, angle=np.pi/2,
                   start_angle=0.0, positive=False)
]

# 4. Build final cross-section
geometry = [top_flange, bottom_flange, web] + fillet_blocks + outer_arcs
cs = CrossSection(geometry=geometry)

# 5. Visualize
ObjectRenderer(CrossSectionGeo(cs, merged=False), 'plotly').show()
ObjectRenderer(CrossSectionGeo(cs), 'plotly').show()

# 6. Extract geometric properties
A = cs.area
y_s = cs.center_of_mass_y
z_s = cs.center_of_mass_z
Iy = cs.mom_of_int
width_cs = cs.width
height_cs = cs.height
(yb, zb) = cs.boundary()
y_bottom, y_top = yb[0], yb[1]
z_top, z_bottom = zb[0], zb[1]

print("=== IPE-like Cross-Section (rounded, approx.) ===")
print(f"Given dims: height={height}, width={width}, tw={tw}, tf={tf}, r={r}")
print(f"Area A                     : {A:.4f}")
print(f"Centroid y_s, z_s         : ({y_s:.4f}, {z_s:.4f})")
print(f"Moment of inertia Iy      : {Iy:.4f}")
print(f"Width, Height             : {width_cs:.4f}, {height_cs:.4f}")
print(f"Boundary in y-direction   : bottom = {y_bottom:.4f},"
      f" top = {y_top:.4f}")
print(f"Boundary in z-direction   : top = {z_top:.4f},"
      f" bottom = {z_bottom:.4f}")
