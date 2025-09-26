
from typing import Any


"""Default styling dictionary for line objects in Plotly.
- mode='lines': draw lines between points.
- line_color='black': default line color.
- showlegend=False: disables legend entry.
- hoverinfo='skip': disables hover tooltips.
"""
DEFAULT_LINE: dict[str, Any] = dict(
    mode='lines',
    line_color='black',
    showlegend=False,
    hoverinfo='skip'
)

"""Default styling dictionary for text labels in Plotly.
- mode='text': displays only text.
- textfont: font size and family.
- showlegend=False: disables legend entry.
"""
DEFAULT_TEXT: dict[str, Any] = dict(
    mode='text',
    textfont={'size': 20, 'family': 'Times New Roman'},
    showlegend=False
)

DEFAULT_SAVE: dict[Any, Any] = dict(
    width=2400,
    height=1600,
    scale=3
)
