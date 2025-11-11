# Cross-Section: H-shaped

from sstatics.core.preprocessing.geometry import Polygon
from sstatics.core.preprocessing import CrossSection
from sstatics.graphic_objects import CrossSectionGraphic

# Outer rectangle (full I-section)
positiv_rectangle = Polygon(
    points=[(50, 100), (50, -100), (-50, -100), (-50, 100), (50, 100)]
)

# Negative rectangles to remove material and create the web
negativ_rectangle_left = Polygon(
    points=[(-2.8, 91.5), (-2.8, -91.5), (-50, -91.5), (-50, 91.5),
            (-2.8, 91.5)],
    positive=False
)
negativ_rectangle_right = Polygon(
    points=[(2.8, 91.5), (50, 91.5), (50, -91.5), (2.8, -91.5), (2.8, 91.5)],
    positive=False
)

# Combine polygons into one cross-section
i_section = CrossSection(
    geometry=[
        positiv_rectangle, negativ_rectangle_left, negativ_rectangle_right
    ]
)

# Visualize the geometry-object that create the cross-section
CrossSectionGraphic(cross_section=i_section, merged=False).show()

# Visualize the cross-section
CrossSectionGraphic(cross_section=i_section).show()
