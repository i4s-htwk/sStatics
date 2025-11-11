# Cross-Section: Beam with a Slab

# Import Moduls
from sstatics.core.preprocessing.geometry import Polygon
from sstatics.core.preprocessing import CrossSection
from sstatics.graphic_objects import CrossSectionGraphic

# Create Geometry

# Slab: 24 units thick
slab = Polygon([(0, 0), (200, 0), (200, 24), (0, 24), (0, 0)])

# Beam: 30 units wide, 60 units high, centered under slab
beam = Polygon([(85, 24), (115, 24), (115, 84), (85, 84), (85, 24)])

# Create Cross-Section

# Combine polygons into one cross-section
beam_with_slab = CrossSection(geometry=[slab, beam])

# Visualize Cross-Section

# Visualize the geometry-object that create the cross-section
CrossSectionGraphic(cross_section=beam_with_slab, merged=False).show()

# Visualize united cross-section

# Visualize the cross-section
CrossSectionGraphic(cross_section=beam_with_slab).show()
