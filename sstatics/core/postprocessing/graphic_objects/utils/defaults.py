
from typing import Any

import numpy as np

DEFAULT = dict(
    showlegend=False,
    hoverinfo='skip'
)

"""Default styling dictionary for line objects in Plotly.
- mode='lines': draw lines between points.
- line_color='black': default line color.
- showlegend=False: disables legend entry.
- hoverinfo='skip': disables hover tooltips.
"""
DEFAULT_LINE: dict[str, Any] = dict(
    mode='lines',
    line_color='black',
    line=dict(width=2),
    **DEFAULT
)

"""Default styling dictionary for text labels in Plotly.
- mode='text': displays only text.
- textfont: font size and family.
- showlegend=False: disables legend entry.
"""
DEFAULT_TEXT: dict[str, Any] = dict(
    mode='text',
    textfont=dict(size=20, family='Times New Roman', color='black'),
    **DEFAULT
)

DEFAULT_TEXT_OFFSET = -1

DEFAULT_POINT = dict(
    mode='markers',
    marker=dict(size=2, color='black'),
    **DEFAULT
)

DEFAULT_CIRCLE_TEXT = dict(
    mode='markers+text',
    textfont=dict(size=20, family='Times New Roman', color='black'),
    marker=dict(
        symbol='circle',
        color='white',
        size=27,
        line=dict(color='black', width=2)
    ),
    **DEFAULT
)

DEFAULT_POLYGON = dict(
    fill='toself',
    fillcolor='rgba(0, 0, 0, 0)'
)

DEFAULT_CENTER_OF_MASS = dict(
    mode='markers',
    marker=dict(size=10),
)

# Mpl
# DEFAULT_TEXT: dict[str, any] = dict(
#     fontsize=20,
#     fontfamily='Times New Roman',
#     color='black',
#     alpha=1.0,
# )
#
# DEFAULT_LINE: dict[str, any] = dict(
#     color='black',
#     linewidth=1,
#     linestyle='-',
#     # alpha=1.0,
# )
#
# DEFAULT_POINT: dict[str, any] = dict(
#     marker='o',
#     markersize=8,
#     markerfacecolor='black',
#     markeredgecolor='black',
#     markeredgewidth=1,
#     alpha=1.0,
# )
#
# DEFAULT_POLYGON: dict[str, any] = dict(
#     facecolor=(0.6, 0.12, 0, 0.1),  # transparent
#     edgecolor='black',
#     linewidth=1,
#     fill=True,
# )
#
# DEFAULT_CENTER_OF_MASS: dict[str, any] = dict(
#     marker='o',
#     markersize=8,
#     markerfacecolor='black',
#     markeredgecolor='black',
#     markeredgewidth=1,
#     # alpha=1.0,
# )

DEFAULT_CROSS_SECTION_POSITIVE = dict(
    fillcolor='rgba(60, 225, 0, 0.1)'
)

DEFAULT_CROSS_SECTION_POINT_STYLE_POSITIVE = dict(
    mode='markers',
    marker=dict(color='rgba(60, 225, 0, 1)'),
)

DEFAULT_CROSS_SECTION_NEGATIVE = dict(
    fillcolor='rgba(255, 0, 0, 0.1)'
)

DEFAULT_CROSS_SECTION_POINT_STYLE_NEGATIVE = dict(
    mode='markers',
    marker=dict(color='rgba(255, 0, 0, 1)'),
)

DEFAULT_HATCH = dict(
    spacing=0.2,
    angle=-np.pi / 4
)

DEFAULT_HINGE = dict(
    line=dict(width=3)
)

DEFAULT_NORMAL_HINGE = dict(
    width=1.0,
    height=0.6
)

DEFAULT_SHEAR_HINGE = dict(
    width=0.25,
    height=1.0
)

DEFAULT_MOMENT_HINGE = dict(
    width=0.5,
    height=0.5
)

DEFAULT_FILL_WHITE = dict(
    fillcolor='white'
)

DEFAULT_SUPPORT = dict(
    line=dict(width=2),
)

DEFAULT_SUPPORT_HATCH = dict(
    spacing=0.36,
)

DEFAULT_CHAMPED_SUPPORT_HATCH = dict(
    spacing=0.36,
    angle=np.pi / 4
)

DEFAULT_FREE_NODE = dict(
    width=0.0,
    height=0.0
)

DEFAULT_ROLLER_SUPPORT = dict(
    width=2.0,
    height=2.0
)

DEFAULT_PINNED_SUPPORT = dict(
    width=3.5,
    height=2.0
)

DEFAULT_FIXED_SUPPORT_UW = dict(
    width=2.0,
    height=2.0
)

DEFAULT_FIXED_SUPPORT_UPHI = dict(
    width=1.0,
    height=2.0
)

DEFAULT_FIXED_SUPPORT_WPHI = dict(
    width=3.0,
    height=2.0
)

DEFAULT_CHAMPED_SUPPORT = dict(
    width=0.5,
    height=2.0
)

DEFAULT_TRANSLATIONAL_SPRING = dict(
    width=2.0,
    height=2.75
)

DEFAULT_TORSIONAL_SPRING = dict(
    width=3.5,
    height=4.0
)

DEFAULT_ARROW = dict(
    line=dict(width=2)
)

DEFAULT_ARROW_HEAD = dict(
    fill='toself',
    fillcolor='black'
)

DEFAULT_LOAD_DISTANCE = 1
DEFAULT_ARROW_DISTANCE = 1
DEFAULT_TENSILE_ZONE_DISTANCE = 0.1

DEFAULT_DISPLACEMENT = dict(
    width_head=0.28,
    length_head=0.4,
    radius=1.0,
    angle_span=(-np.pi / 4, -3 * np.pi / 4)
)

DEFAULT_POINT_FORCE = dict(
    width_head=0.28,
    length_head=0.8,
    length_tail=2.2,
)

DEFAULT_POINT_MOMENT = dict(
    width_head=0.28,
    length_head=0.4,
    radius=1.3,
    angle_span=(np.pi / 6, -np.pi / 4)
)

DEFAULT_BAR = dict(
    line=dict(width=4),
)

DEFAULT_TENSILE_ZONE = dict(
    line=dict(dash='dash', width=1),
)

DEFAULT_STATE_LINE = dict(
    line_color='red'
)

DEFAULT_STATE_LINE_TEXT = dict(
    textfont=dict(color='red')
)

PLOTLY = 'plotly'
MPL = 'mpl'
VALID_MODES = (PLOTLY, MPL)
DEFAULT_MODE = PLOTLY

DEFAULT_LAYOUT_X = dict(
    autorange=True
)

DEFAULT_LAYOUT_Y = dict(
    autorange='reversed',
    scaleanchor='x',
    scaleratio=1
)

DEFAULT_NUMBER_OF_TEXT_POSITIONS = 4
DEFAULT_NUMBER_OF_TEXT_RINGS = 2

DEFAULT_SAVE: dict[Any, Any] = dict(
    width=2400,
    height=1600,
    scale=3
)
