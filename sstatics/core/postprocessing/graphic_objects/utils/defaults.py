
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

DEFAULT_CROSS_SECTION_POSITIVE = 'rgba(60, 225, 0, 0.1)'

DEFAULT_CROSS_SECTION_NEGATIVE = 'rgba(255, 0, 0, 0.1)'

DEFAULT_HATCH = dict(
    spacing=0.2,
    angle=-np.pi / 4
)

DEFAULT_HINGE = dict(
    line=dict(width=3)
)

DEFAULT_NORMAL_HINGE = dict(
    width=11 / 20,
    height=11 / 30
)

DEFAULT_SHEAR_HINGE = dict(
    width=11 / 80,
    height=11 / 20
)

DEFAULT_MOMENT_HINGE = dict(
    width=11 / 40,
    height=11 / 40
)

DEFAULT_FILL_WHITE = dict(
    fillcolor='rgba(255, 255, 255, 1)'
)

PLOTLY = 'plotly'
MPL = 'mpl'
VALID_MODES = (PLOTLY, MPL)
DEFAULT_MODE = PLOTLY

DEFAULT_SAVE: dict[Any, Any] = dict(
    width=2400,
    height=1600,
    scale=3
)
