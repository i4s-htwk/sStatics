import re

import matplotlib.colors as mcolors


LINESTYLE_MAP = {
    'solid': '-',
    'dot': ':',
    'dash': '--',
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
        f"Stil konnte nicht erkannt oder konvertiert werden: {style}"
    )


def convert_plotly_to_mpl(style: dict) -> dict:

    mpl_style = {}

    if style['mode'] == 'markers' and 'marker' in style:
        mpl_style['marker'] = 'o'
        sm = style['marker']
        if 'size' in sm:
            mpl_style['markersize'] = sm['size'] ** 0.5 * 3
        if 'color' in sm:
            mpl_style['markerfacecolor'] = convert_color_to_mpl(sm['color'])
        if 'line' in sm:
            sml = sm['line']
            if 'color' in sml:
                mpl_style['markeredgecolor'] = convert_color_to_mpl(
                    sml['color']
                )
            if 'width' in sml:
                mpl_style['markeredgewidth'] = sml['width'] * 2 / 3
        if 'opacity' in sm:
            mpl_style['alpha'] = sm['opacity']

    # Text (vielleicht eher .get() verwenden?, dann keine if's)
    if style['mode'] == 'text':
        if 'textfont' in style:
            st = style['textfont']
            if 'size' in st:
                mpl_style['fontsize'] = st['size']
            if 'color' in st:
                mpl_style['color'] = convert_color_to_mpl(st['color'])
            if 'family' in st:
                mpl_style['fontfamily'] = st['family']
        if 'opacity' in style:
            mpl_style['alpha'] = style['opacity']

    if 'line_color' in style and 'fillcolor' not in style:
        mpl_style['color'] = convert_color_to_mpl(style['line_color'])
    elif 'line_color' in style:
        mpl_style['edgecolor'] = convert_color_to_mpl(style['line_color'])

    if 'line' in style:
        sl = style['line']
        if 'width' in sl:
            mpl_style['linewidth'] = sl['width']
        if 'dash' in sl:
            if sl['dash'] not in LINESTYLE_MAP:
                raise ValueError(f'Unrecognized line dash style: {sl['dash']}')
            mpl_style['linestyle'] = LINESTYLE_MAP[sl['dash']]

    if 'fillcolor' in style:
        mpl_style['fill'] = True
        mpl_style['facecolor'] = convert_color_to_mpl(style['fillcolor'])

    if 'opacity' in style:
        mpl_style['alpha'] = style['opacity']

    return mpl_style


def convert_color_to_mpl(color: str | tuple | list | None):
    """
    Konvertiert Plotly-Farbangaben in Matplotlib-kompatible RGBA-Werte (0–1).

    Unterstützte Eingabeformate:
      - Hex: "#RRGGBB" oder "#RRGGBBAA"
      - RGB/RGBA: "rgb(r, g, b)" / "rgba(r, g, b, a)"
      - CSS-Farbnamen (z.B. "red", "skyblue")
      - Plotly-kompatible Tupel oder Listen (z.B. [255, 0, 0],
      [255, 0, 0, 0.5])
      - Matplotlib-kompatible Tupel (0–1)
    """

    if color is None:
        return None

    # ---- 1️⃣ Hex-Codes (#RRGGBB oder #RRGGBBAA)
    if isinstance(color, str) and color.startswith("#"):
        try:
            return mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f"Ungültiger Hex-Farbcode: {color!r}")

    # ---- 2️⃣ CSS-/Plotly-Strings ("rgb(...)", "rgba(...)")
    if isinstance(color, str) and color.lower().startswith(
            ("rgb", "rgba")):
        match = re.match(
            r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d\.]+))?\)",
            color.strip().lower()
        )
        if match:
            r, g, b = [int(match.group(i)) / 255 for i in range(1, 4)]
            a = float(match.group(4)) if match.group(
                4) is not None else 1.0
            return (r, g, b, a)
        raise ValueError(f"Ungültige RGB/RGBA-Farbangabe: {color!r}")

    # ---- 3️⃣ CSS-/Matplotlib-Farbnamen (z. B. "red", "steelblue", "black")
    if isinstance(color, str):
        try:
            return mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f"Unbekannter Farbname: {color!r}")

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
            raise ValueError(f"Ungültige Farblänge: {color!r}")
        return color

    raise TypeError(
        f"Nicht unterstützter Farbtyp: {type(color).__name__} → {color!r}")


def convert_mpl_to_plotly(style: dict) -> dict:
    plotly_style = {}

    # Linien (Standard)
    line_dict = {}
    if "linewidth" in style:
        line_dict["width"] = style["linewidth"]

    if "linestyle" in style:
        linestyle = style["linestyle"]
        reverse_linestyle_map = {v: k for k, v in LINESTYLE_MAP.items()}
        if linestyle in reverse_linestyle_map:
            line_dict["dash"] = reverse_linestyle_map[linestyle]
        else:
            line_dict["dash"] = "solid"

    # Farbe: kann facecolor, edgecolor oder color heißen
    if "facecolor" in style:
        plotly_style["fillcolor"] = convert_color_to_plotly(style["facecolor"])
        plotly_style['fill'] = 'toself'
    if "edgecolor" in style:
        plotly_style["line_color"] = convert_color_to_plotly(
            style["edgecolor"]
        )
    elif "color" in style:
        plotly_style["line_color"] = convert_color_to_plotly(style["color"])

    # Deckkraft
    if "alpha" in style:
        plotly_style["opacity"] = style["alpha"]

    # Marker
    if "marker" in style or "markersize" in style:
        marker_dict = {}
        if "markersize" in style:
            marker_dict["size"] = (style["markersize"] / 3) ** 2
        if "markerfacecolor" in style:
            marker_dict["color"] = convert_color_to_plotly(
                style["markerfacecolor"]
            )
        if "markeredgecolor" in style:
            marker_dict.setdefault("line", {})["color"] \
                = convert_color_to_plotly(style["markeredgecolor"])
        if "markeredgewidth" in style:
            marker_dict.setdefault("line", {})["width"] \
                = style["markeredgewidth"]
        plotly_style["marker"] = marker_dict
        plotly_style["mode"] = "markers"

    # Text
    if any(k in style for k in ("fontsize", "fontfamily")):
        textfont = {}
        if "fontsize" in style:
            textfont["size"] = style["fontsize"]
        if "color" in style:
            textfont["color"] = convert_color_to_plotly(style["color"])
        if "fontfamily" in style:
            textfont["family"] = style["fontfamily"]
        plotly_style["textfont"] = textfont
        plotly_style["mode"] = "text"

    # Linieninformationen hinzufügen, falls vorhanden
    if line_dict:
        plotly_style["line"] = line_dict
        if "line_color" not in plotly_style:
            plotly_style["line_color"] = convert_color_to_plotly(
                style.get("color", "black")
            )
        if "mode" not in plotly_style:
            plotly_style["mode"] = "lines"

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
            raise ValueError(f"Ungültige Farbdefinition: {color!r}")
    elif isinstance(color, (tuple, list)):
        rgba = color
    else:
        raise TypeError(f"Ungültiger Farbtyp: {type(color).__name__}")

    # Sicherstellen, dass es 4 Komponenten gibt
    if len(rgba) == 3:
        rgba = (*rgba, 1.0)

    r, g, b = [int(round(255 * c)) for c in rgba[:3]]
    a = float(rgba[3])
    return f"rgba({r}, {g}, {b}, {a:.3f})"
