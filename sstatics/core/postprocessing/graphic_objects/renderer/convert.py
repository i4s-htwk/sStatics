
import re

import matplotlib.colors as mcolors


LINESTYLE_MAP = {
    'solid': '-',
    'dot': ':',
    'dash': (10, 6),
    'longdash': (0, (8, 4)),
    'dashdot': '-.',
}


def detect_style_type(style: dict) -> str:
    """
    Erkennt automatisch, ob ein Stil Plotly- oder Matplotlib-Struktur hat.
    Gibt 'plotly', 'mpl' oder 'unknown' zurück.
    """
    plotly_keys = {'mode', 'line', 'fillcolor', 'line_color', 'textfont'}
    mpl_keys = {'color', 'linewidth', 'linestyle', 'facecolor', 'edgecolor',
                'fontsize', 'markersize', 'markerfacecolor', 'markeredgecolor',
                'markeredgewidth'}

    # Wenn eindeutig Plotly-Schlüssel enthalten sind
    if any(k in style for k in plotly_keys):
        return 'plotly'

    # Wenn eindeutig Matplotlib-Schlüssel enthalten sind
    if any(k in style for k in mpl_keys):
        return 'mpl'

    # Falls der Stil ein verschachteltes Objekt enthält
    # (z. B. style['marker'] = {...})
    if 'marker' in style and isinstance(style['marker'], dict):
        # Plotly benutzt marker=dict(...), mpl nicht
        return 'plotly'

    return 'unknown'


def convert_style(style: dict, target: str) -> dict:
    """
    Wandelt automatisch Plotly ⇄ Matplotlib um, je nach Ziel 'mpl' oder
    'plotly'.
    """
    style_type = detect_style_type(style)

    if style_type == target:
        return style  # nichts zu tun

    if target == 'mpl' and style_type == 'plotly':
        return convert_plotly_to_mpl(style)

    if target == 'plotly' and style_type == 'mpl':
        return convert_mpl_to_plotly(style)

    raise ValueError(
        f'Stil konnte nicht erkannt oder konvertiert werden: {style}'
    )


def convert_plotly_to_mpl(style: dict) -> dict:
    """
    Convert a Plotly style dict to a Matplotlib-compatible style dict.
    Supports 'markers', 'text', and 'markers+text'.
    """
    mpl_style = {}

    mode = style.get('mode', '')

    # --- MARKER ---
    if 'markers' in mode and 'marker' in style:
        sm = style['marker']
        mpl_style['marker'] = 'o'
        if 'size' in sm:
            mpl_style['markersize'] = sm['size'] ** 0.5 * 4
        if 'color' in sm:
            mpl_style['markerfacecolor'] = convert_color_to_mpl(sm['color'])
        sml = sm.get('line', {})
        if 'color' in sml:
            mpl_style['markeredgecolor'] = convert_color_to_mpl(sml['color'])
        if 'width' in sml:
            mpl_style['markeredgewidth'] = sml['width'] * 2 / 3
        if 'opacity' in sm:
            mpl_style['alpha'] = sm['opacity']

    # --- TEXT ---
    if 'text' in mode:
        st = style.get('textfont', {})
        if 'size' in st:
            mpl_style['fontsize'] = st['size']
        if 'color' in st:
            mpl_style['color'] = convert_color_to_mpl(st['color'])
        if 'family' in st:
            mpl_style['fontfamily'] = st['family']
        if 'opacity' in style:
            mpl_style['alpha'] = style['opacity']

    # --- LINES ---
    line_color = style.get('line_color')
    if line_color and not style.get('fillcolor'):
        mpl_style['color'] = convert_color_to_mpl(line_color)
    elif line_color:
        mpl_style['edgecolor'] = convert_color_to_mpl(line_color)

    sl = style.get('line', {})
    if 'width' in sl:
        mpl_style['linewidth'] = sl['width']
    if 'dash' in sl:
        dash = sl['dash']
        if dash not in LINESTYLE_MAP:
            raise ValueError(f'Unrecognized line dash style: {dash}')

        base_pattern = LINESTYLE_MAP[dash]
        scale = sl.get('dash_scale', 1.0)  # optionaler Skalierungsfaktor
        mpl_style['dashes'] = [x * scale for x in base_pattern]
        mpl_style['linestyle'] = (0, mpl_style['dashes'])

    # --- FILL ---
    if 'fillcolor' in style:
        mpl_style['fill'] = True
        mpl_style['facecolor'] = convert_color_to_mpl(style['fillcolor'])

    # --- GLOBAL OPACITY ---
    if 'opacity' in style:
        mpl_style['alpha'] = style['opacity']

    return mpl_style


def convert_color_to_mpl(color: str | tuple | list | None):
    """
    Konvertiert Plotly-Farbangaben in Matplotlib-kompatible RGBA-Werte (0–1).

    Unterstützte Eingabeformate:
      - Hex: '#RRGGBB' oder '#RRGGBBAA'
      - RGB/RGBA: 'rgb(r, g, b)' / 'rgba(r, g, b, a)'
      - CSS-Farbnamen (z.B. 'red', 'skyblue')
      - Plotly-kompatible Tupel oder Listen (z.B. [255, 0, 0],
      [255, 0, 0, 0.5])
      - Matplotlib-kompatible Tupel (0–1)
    """

    if color is None:
        return None

    # ---- 1️⃣ Hex-Codes (#RRGGBB oder #RRGGBBAA)
    if isinstance(color, str) and color.startswith('#'):
        try:
            return mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f'Ungültiger Hex-Farbcode: {color!r}')

    # ---- 2️⃣ CSS-/Plotly-Strings ('rgb(...)', 'rgba(...)')
    if isinstance(color, str) and color.lower().startswith(
            ('rgb', 'rgba')):
        match = re.match(
            r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d\.]+))?\)',
            color.strip().lower()
        )
        if match:
            r, g, b = [int(match.group(i)) / 255 for i in range(1, 4)]
            a = float(match.group(4)) if match.group(
                4) is not None else 1.0
            return (r, g, b, a)
        raise ValueError(f'Ungültige RGB/RGBA-Farbangabe: {color!r}')

    # ---- 3️⃣ CSS-/Matplotlib-Farbnamen (z. B. 'red', 'steelblue', 'black')
    if isinstance(color, str):
        try:
            return mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f'Unbekannter Farbname: {color!r}')

    # ---- 4️⃣ Tupel oder Liste (z. B. [255, 0, 0], [1.0, 0.0, 0.0, 0.5])
    if isinstance(color, (tuple, list)):
        # Falls Integerwerte → normalisieren
        if all(isinstance(c, int) for c in color):
            color = [c / 255 for c in color]
        # Fehlendes Alpha ergänzen
        if len(color) == 3:
            color = tuple(color) + (1.0,)
        elif len(color) == 4:
            color = tuple(color)
        else:
            raise ValueError(f'Ungültige Farblänge: {color!r}')
        return color

    raise TypeError(
        f'Nicht unterstützter Farbtyp: {type(color).__name__}, {color!r}')


def convert_mpl_to_plotly(style: dict) -> dict:
    plotly_style = {}

    # Linien (Standard)
    line_dict = {}
    if 'linewidth' in style:
        line_dict['width'] = style['linewidth']

    if 'linestyle' in style:
        linestyle = style['linestyle']
        reverse_linestyle_map = {v: k for k, v in LINESTYLE_MAP.items()}
        if linestyle in reverse_linestyle_map:
            line_dict['dash'] = reverse_linestyle_map[linestyle]
        else:
            line_dict['dash'] = 'solid'

    # Farbe: kann facecolor, edgecolor oder color heißen
    if 'facecolor' in style:
        plotly_style['fillcolor'] = convert_color_to_plotly(style['facecolor'])
        plotly_style['fill'] = 'toself'
    if 'edgecolor' in style:
        plotly_style['line_color'] = convert_color_to_plotly(
            style['edgecolor']
        )
    elif 'color' in style:
        plotly_style['line_color'] = convert_color_to_plotly(style['color'])

    # Deckkraft
    if 'alpha' in style:
        plotly_style['opacity'] = style['alpha']

    # Marker
    if 'marker' in style or 'markersize' in style:
        marker_dict = {}
        if 'markersize' in style:
            marker_dict['size'] = (style['markersize'] / 3) ** 2
        if 'markerfacecolor' in style:
            marker_dict['color'] = convert_color_to_plotly(
                style['markerfacecolor']
            )
        if 'markeredgecolor' in style:
            marker_dict.setdefault('line', {})['color'] \
                = convert_color_to_plotly(style['markeredgecolor'])
        if 'markeredgewidth' in style:
            marker_dict.setdefault('line', {})['width'] \
                = style['markeredgewidth']
        plotly_style['marker'] = marker_dict
        plotly_style['mode'] = 'markers'

    # Text
    if any(k in style for k in ('fontsize', 'fontfamily')):
        textfont = {}
        if 'fontsize' in style:
            textfont['size'] = style['fontsize']
        if 'color' in style:
            textfont['color'] = convert_color_to_plotly(style['color'])
        if 'fontfamily' in style:
            textfont['family'] = style['fontfamily']
        plotly_style['textfont'] = textfont
        plotly_style['mode'] = 'text'

    # Linieninformationen hinzufügen, falls vorhanden
    if line_dict:
        plotly_style['line'] = line_dict
        if 'line_color' not in plotly_style:
            plotly_style['line_color'] = convert_color_to_plotly(
                style.get('color', 'black')
            )
        if 'mode' not in plotly_style:
            plotly_style['mode'] = 'lines'

    return plotly_style


def convert_color_to_plotly(color):
    """
    Konvertiert Matplotlib-kompatible Farben (z. B. RGBA-Tupel oder Farbnamen)
    in Plotly-kompatible Formate (z. B. 'rgba(r,g,b,a)').
    """

    if color is None:
        return None

    # String → direkt zurück (z. B. 'red', '#00ff00')
    if isinstance(color, str):
        try:
            rgba = mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f'Ungültige Farbdefinition: {color!r}')
    elif isinstance(color, (tuple, list)):
        rgba = color
    else:
        raise TypeError(f'Ungültiger Farbtyp: {type(color).__name__}')

    # Sicherstellen, dass es 4 Komponenten gibt
    if len(rgba) == 3:
        rgba = (*rgba, 1.0)

    r, g, b = [int(round(255 * c)) for c in rgba[:3]]
    a = float(rgba[3])
    return f'rgba({r}, {g}, {b}, {a:.3f})'


def convert_plotly_to_mpl_layout(x_opts: dict, y_opts: dict):

    mpl_x, mpl_y = {}, {}

    if 'xlim' in x_opts and isinstance(x_opts['xlim'], (tuple, list)):
        mpl_x['xlim'] = list(x_opts['xlim'])
    elif 'range' in x_opts and isinstance(x_opts['range'], (tuple, list)):
        mpl_x['xlim'] = list(x_opts['range'])

    if 'ylim' in y_opts and isinstance(y_opts['ylim'], (tuple, list)):
        mpl_y['ylim'] = list(y_opts['ylim'])
    elif 'range' in y_opts and isinstance(y_opts['range'], (tuple, list)):
        mpl_y['ylim'] = list(y_opts['range'])

    if 'invert_yaxis' in y_opts:
        mpl_y['invert_yaxis'] = y_opts['invert_yaxis']
    elif y_opts.get('autorange') == 'reversed':
        mpl_y['invert_yaxis'] = True

    if y_opts.get('aspect') == 'equal':
        mpl_y['aspect'] = 'equal'
    elif y_opts.get('scaleanchor') == 'x' and y_opts.get('scaleratio') == 1:
        mpl_y['aspect'] = 'equal'

    return mpl_x, mpl_y


def convert_mpl_to_plotly_layout(x_opts: dict, y_opts: dict):

    plotly_x, plotly_y = {}, {}

    if 'xlim' in x_opts and isinstance(x_opts['xlim'], (tuple, list)):
        plotly_x['range'] = list(x_opts['xlim'])
    elif 'range' in x_opts and isinstance(x_opts['range'], (tuple, list)):
        plotly_x['range'] = list(x_opts['range'])

    if x_opts.get('invert_xaxis'):
        plotly_x['autorange'] = 'reversed'
    elif 'autorange' in x_opts:
        plotly_x['autorange'] = x_opts['autorange']

    if 'ylim' in y_opts and isinstance(y_opts['ylim'], (tuple, list)):
        plotly_y['range'] = list(y_opts['ylim'])
    elif 'range' in y_opts and isinstance(y_opts['range'], (tuple, list)):
        plotly_y['range'] = list(y_opts['range'])

    if y_opts.get('invert_yaxis'):
        plotly_y['autorange'] = 'reversed'
    elif 'autorange' in y_opts:
        plotly_y['autorange'] = y_opts['autorange']

    if y_opts.get('aspect') == 'equal':
        plotly_y['scaleanchor'] = 'x'
        plotly_y['scaleratio'] = 1
    else:
        if 'scaleanchor' in y_opts:
            plotly_y['scaleanchor'] = y_opts['scaleanchor']
        if 'scaleratio' in y_opts:
            plotly_y['scaleratio'] = y_opts['scaleratio']

    return plotly_x, plotly_y


def convert_text_style_for_annotation(style: dict) -> dict:
    """
    Convert a Scatter text style dict to an Annotation-compatible dict.
    """
    ann_style = {}

    # font dict
    font = style.get("textfont", {})
    ann_style["font"] = dict(
        size=font.get("size", 12),
        color=font.get("color", "black"),
        family=font.get("family", "Arial")
    )

    # Anchoring defaults
    ann_style.setdefault("xanchor", "center")
    ann_style.setdefault("yanchor", "middle")
    ann_style.setdefault("showarrow", False)

    return ann_style
