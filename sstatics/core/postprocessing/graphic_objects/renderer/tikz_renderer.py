from functools import cached_property
import numpy as np
from .base_renderer import AbstractRenderer
from sstatics.core.postprocessing.graphic_objects.utils.defaults import TIKZ


class TikzRenderer(AbstractRenderer):
    """
    Renderer, der ein statisches System als TikZ-Code (.tex-String)
    ausgibt und es grafisch anzeigen kann.
    """
    # übernimmt dabei die Funktionen von AbstractRenderer

    SCALE_MACRO_NAME = 'scale'                      # Name der Variable für die Skalierung in TikZ
    SCALE_DEFAULT_VALUE = 2.0                       # Standard-Skalierung für TikZ (1.0 = 100%)
    LOAD_SCALE_MACRO_NAME = 'loadscale'             # eigene Variable nur für die Pfeillänge der Lasten
    LOAD_SCALE_DEFAULT_VALUE = 1.0                  # Startwert, unabhängig von \scale
    SYMBOL_SCALE_MACRO_NAME = 'symbolscale'         # eigene Variable nur für Auflager-/Gelenkgrößen
    SYMBOL_SCALE_DEFAULT_VALUE = 1.0                # unabhängig von \scale
    LOAD_LABEL_FONT_SIZE = 14                       # Schriftgröße für Lastbeschriftungen
    LOAD_ANGLE_OFFSET = 0.0                         # Kalibrierungs-Offset, siehe Erklärung unten
    LOAD_LABEL_DISTANCE = 0.9                       # fester Abstand (Modelleinheiten) der Einzellast-/Reaktionsbeschriftung, unabhängig von der Systemgröße
    LINELOAD_LABEL_DISTANCE = 1.0                   # fester Abstand der Linienlast-Beschriftung von der Lastlinie
    LINELOAD_LABEL_SIDE = 1                         # +1 oder -1, falls die Beschriftung auf der falschen Seite landet, hier umdrehen
    LINELOAD_DISTANCE_MM = 3.0                      # Abstand der Linienlastbeschriftung in mm
    LINELOAD_LABEL_CLEARANCE_MM = 10.0             # Beschriftungsabstand von selbst gezeichneten Linienlasten in mm
    AXIS_TICK_STEP = 0.5                            # Beschriftungsabstand der Achsen (in realen Einheiten)
    AXIS_MARGIN_TICKS = 2                           # wie viele Ticks Platz rundherum
    SF_SCALE_MACRO_PREFIX = 'sfscale'               # Skalierung für pro-Fläche-Makros (sfscale1, sfscale2, ...)
    SF_SCALE_DEFAULT_VALUE = 20                     # Startwert, für alle Flächen identisch, unabhängig von \scale
    SF_LABEL_ANCHORS = ('below right', 'below left', 'below')  # Anker für (Anfang, Ende, Extremum)
    MOMENT_ANGLES_POSITIVE = (110.0, 115.0)         # (Startwinkel, Aufspannwinkel) in °, positives Moment: linke Seite, im Uhrzeigersinn
    MOMENT_ANGLES_NEGATIVE = (315, 115.0)           # dito für negatives Moment: rechte Seite, gegen den Uhrzeigersinn!


    def __init__(
            self,
            show_axis: bool = True,             # Achsen in der Vorschau anzeigen
            show_grid: bool = True,             # anzeige des Gitternetzes in der Vorschau ???
            x_opts: dict | None = None,
            y_opts: dict | None = None,
            show_code: bool = True,             # TikZ-Code in der Terminal-Konsole ausgeben
            show_pdf: bool = True,              # PDF-Generierung und Anzeige
            show_support_labels: bool = False,   # Anzeige der Knotenbezeichnungen
            show_bar_numbers: bool = True,      # Anzeige der Stabnummern
            show_dimensioning: bool = True,
            show_loads: bool = True,            # Anzeige der Lasten
            show_reactions: bool = False,          # Anzeige der Auflagerkräfte
            save_tikz: bool = False,            # als reine .tikz-Datei speichern (nur tikzpicture, ohne Header)
            tikz_filename: str | None = None,   # eigener Dateiname für die .tikz-Datei
            save_tex: bool = False,             # als .tex-Datei automatisch speichern
            tex_output_dir: str | None = None,  # Zielordner der Speicherung, None = Downloads-Ordner
            tex_filename: str | None = None,    # eigener Dateiname, um mehrere TeX-Dateien erzeugen zu können
            **kwargs
    ):
        super().__init__(mode=TIKZ)             # stellt Aufbau von AbstractRenderer sicher
        self._show_axis = show_axis
        self._show_grid = show_grid
        self._show_code = show_code
        self._show_pdf = show_pdf
        self._show_support_labels = show_support_labels   
        self._show_bar_numbers = show_bar_numbers 
        self._show_dimensioning = show_dimensioning
        self._show_loads = show_loads
        self._show_reactions = show_reactions
        self._save_tikz = save_tikz                  # als reine .tikz-Datei automatisch speichern
        self._tikz_filename = tikz_filename
        self._save_tex = save_tex                    # als .tex-Datei automatisch speichern
        self._tex_output_dir = tex_output_dir
        self._tex_filename = tex_filename
        self._draw_commands = []                     # reine \draw-Befehle
        self._text_node_specs = []                   # Spezifikationen für Textknoten
        self._point_commands = []
        self._point_names = {}
        self._point_counter = 0
        self._beam_commands = []
        self._load_commands = []
        self._load_keys_seen = set()
        self._reaction_commands = []                 # Liste für Auflagerkräfte
        self._reaction_keys_seen = set()             # Dopplungs-Set (Kraft)
        self._reaction_moment_keys_seen = set()      # Dopplungs-Set (Moment)
        self._support_commands = []
        self._support_keys_seen = set()
        self._hinge_commands = []
        self._hinge_keys_seen = set()
        self._lineload_commands = []
        self._lineload_keys_seen = set()        # \lineload-Befehl statt einzelne Linien und Pfeile
        self._moment_keys_seen = set()
        self._dimensioning_commands = []
        self._static_force_commands = []        # Eigene Liste für Schnittkraftflächen (Trennung von Zeichenbefehlen)
        self._color_names = {}                  # merkt bereits definierte Farben (RGBA -> RGB)
        self._color_defs = []                   # Liste der TikZ-Farbdefinitionen
        self._color_counter = 0                 # Zähler für eindeutige Farbnamen (sscolor1, sscolor2, ...)
        self._font_size_names = {}              # wie self._color_counter, nur für Schriftgrößen-Variablen (fsz1, fsz2, ...)
        self._font_size_defs = []               # Liste der TikZ-Schriftgrößendefinitionen
        self._sf_scale_names = {}               # sf_index -> Makroname (sfscale1, sfscale2, ...)
        self._sf_scale_defs = []                # Liste der TikZ-Makrodefinitionen für die Schnittkraftflächen-Skalierung
        
    def _layout(self):
        pass

    def _fmt_scaled_value(self, value):                                     # für die Skalierung der Dimensionierung mit stanli
        """Skaliert einen einzelnen Zahlenwert (keine Koordinate) mit \\scale.
        Wird für den distance-Parameter von \\dimensioning gebraucht, der
        nur EIN Zahlenwert ist, kein Koordinatenpaar wie bei _fmt_coord."""
        macro = self.SCALE_MACRO_NAME
        return f'\\{macro}*{value:.4f}'
    
    def _fmt_scaled_coord_parts(self, x, z):
        """Skaliert die Koordinaten mit der vorgegebenen Variable SCALE_MACRO_NAME und gibt sie einzeln zurück.
        Keine Kollision mit _fmt_coord, da diese Methode nur die Teile zurückgibt, während _fmt_coord die Koordinaten in TikZ-Format zusammenfügt.
        So können für die statische Analyse Bibliothek die Koordinaten auch einzeln in die TikZ-Befehle integriert werden."""
        macro = self.SCALE_MACRO_NAME
        x_str = f'\\{macro}*{x:.4f}'
        z_str = f'\\{macro}*{-z:.4f}'           # z-Achse invertiert für TikZ
        return x_str,z_str
    
    def _fmt_coord(self, x, z):
        """Skaliert das gesamte System mit der vorgegebenen Variable SCALE_MACRO_NAME und gibt die Koordinaten mit 4 Nachkommastellen zurück.
        Dies ist erforderlich um in TikZ für Befehle wie \draw oder \node die Koordinaten korrekt zu übergeben."""
        x_str, z_str = self._fmt_scaled_coord_parts(x, z)      
        return f'({x_str},{z_str})'                # gibt die Koordinaten in TikZ-Format zurück, z.B. (\\scale*1.0000,\\scale*-2.0000)
    
    def _symbol_scale_overrides(self):
        """Überschreibt die in structuralanalysis.sty fest verdrahteten Längenkonstanten für Auflager- und Gelenksymbole, sodass sie mit SYMBOL_SCALE_MACRO_NAME skaliert werden können. 
        Die Bibliothek multipliziert diese Werte nirgends automatisch mit irgendeiner Skalierungsvariable und \\scale wirkt nur auf Koordinaten (\\point), \\loadscale nur auf Lastpfeile. 
        Ohne dieses Überschreiben blieben Auflager/Gelenke immer in ihrer festen Originalgröße.
        Die Original-Werte stammen direkt aus structuralanalysis.sty."""
        macro = self.SYMBOL_SCALE_MACRO_NAME
        originals = {
            'supportBasicLength': '6mm',
            'supportBasicHeight': '1mm',
            'supportLength': '4mm',
            'supportHeight': '2.5mm',
            'supportHatchingLength': '10mm',
            'supportHatchingHeight': '1mm',
            'supportGap': '1mm',
            'rollRadius': '0.6mm',
            'hingeRadius': '1mm',
            'hingeSmallRadius': '0.8mm',
            'hingeLargeRadius': '1.2mm',
        }
        return [
            f'\\renewcommand{{\\{name}}}{{{value}*\\{macro}}}'
            for name, value in originals.items()
        ]
    
    def _register_point(self, x, z):
        """Registriert einen Koordinatenpunkt für stanli (\\point{name}{x}{y}).
        Existiert an dieser Stelle bereits ein Punkt, wird kein neuer angelegt,
        sondern der vorhandene Name zurückgegeben."""
        key = (round(x, 4), round(z, 4))            # Rundet die Koordinaten auf 4 Nachkommastellen, um gleiche Punkte zu erkennen
        if key not in self._point_names:
            self._point_counter += 1
            name = f'p{self._point_counter}'
            self._point_names[key] = name
            x_str, z_str = self._fmt_scaled_coord_parts(x, z)   
            self._point_commands.append(
                f'\\point{{{name}}}{{{x_str}}}{{{z_str}}};'
            )
        return self._point_names[key]
    
    def _add_beam(self, x, z, beam_type=1):                                    # Erzeugt einen Stab mit der stanli-Methode \beam{type}{start_point}{end_point}
        """Erzeugt (falls nötig) die \\point-Befehle für Start- und Endpunkt
        und daraus einen \\beam-Befehl von stanli."""
        name_i = self._register_point(x[0], z[0])
        name_j = self._register_point(x[1], z[1])
        self._beam_commands.append(f'\\beam{{{beam_type}}}{{{name_i}}}{{{name_j}}};')
    
    @property
    def _load_label_font_pt(self):
        """Rechnet LOAD_LABEL_FONT_SIZE exakt wie convert_plotly_to_tikz von Plotly-Fontgröße in TikZ-pt um (Faktor 0.75), damit alle Lastwerte (Einzellast, Reaktion, Moment, Linienlast) exakt dieselbe Größe haben."""
        return self.LOAD_LABEL_FONT_SIZE * 0.75
    
    def _fmt_value_node(self, x_str, z_str, value, anchor='above'):     # NEU
            """Hilfsmethode zum Erzeugen eines TikZ-Nodes für einen Last-/Momentwert, damit Lastbeschriftungen alle gleich groß erscheinen."""
            fsz_name, bsz_name = self._get_font_size_names(self._load_label_font_pt)
            font = f'font=\\fontsize{{\\{fsz_name} pt}}{{\\{bsz_name} pt}}\\selectfont'
            return f'\\node[{anchor}, {font}] at ({x_str},{z_str}) {{${value}$}};'
        
    def _scaled_offset_coord(self, x, z, disp_x=0.0, disp_z=0.0):
        """Baut die skalierten TikZ-Koordinatenteile für einen Punkt (x, z) in Modellkoordinaten, zu dem zusätzlich eine bereits in Bildschirmrichtung (also NACH der z-Spiegelung) gegebene Verschiebung (disp_x, disp_z) addiert wird. 
        So kann man Punkte um einen festen Betrag verschieben, ohne dass _base_scale (Systemgröße) mit hineinspielt."""
        macro = self.SCALE_MACRO_NAME
        x_str = f'\\{macro}*{x + disp_x:.4f}'
        z_str = f'\\{macro}*{-z + disp_z:.4f}'
        return x_str, z_str

    def _label_offset_point(self, origin, rotation_deg, distance):
        """Berechnet einen Punkt, der von 'origin' aus um 'distance' (feste Modelleinheit) in die durch rotation_deg vorgegebene Richtung verschoben ist.
        Dies ist dieselbe Richtung, in die auch der zugehörige \\load-Pfeil zeigt. 
        Damit ist die Beschriftung immer im gleichen Abstand zum Symbol, egal wie groß das System ist."""
        angle_rad = np.radians(rotation_deg)
        disp_x = distance * np.cos(angle_rad)
        disp_z = distance * np.sin(angle_rad)
        return self._scaled_offset_coord(origin[0], origin[1], disp_x, disp_z)
    
    def _add_load(self, x, z, style):
        """Erzeugt einen \\load-Befehl von stanli für eine Einzelkraft.
        Wird NUR beim Pfeilschaft (zwei Punkte) aufgerufen, nicht beim Pfeilkopf.
        Der Pfeilkopf-Aufruf tut nichts und \load bringt seinen eigenen Kopf ohnehin selbst mit."""
        origin = style.get('load_origin')
        value = style.get('load_value', '')
        if origin is None:
            return

        key = (round(origin[0], 4), round(origin[1], 4), style.get('load_axis'))
        if key in self._load_keys_seen:      # schon erzeugt, daher abbrechen
            return
        self._load_keys_seen.add(key)

        # Richtung aus den zwei (bereits transformierten!) Endpunkten des Pfeilschafts berechnen. 
        # x[0],z[0] liegt näher am Pfeilkopf, x[1],z[1] ist das weit entfernte Ende. 
        # Die Pfeilrichtung zeigt vom entfernten zum nahen Punkt.
        dx = x[0] - x[1]
        dz = z[0] - z[1]
        # -dz wegen invertierter TikZ-z-Achse, +180 weil stanlis \load bei Winkel 0 nach links statt nach rechts zeigt
        angle_deg = np.degrees(np.arctan2(-dz, dx)) + 180
        rotation_deg = (angle_deg + self.LOAD_ANGLE_OFFSET) % 360
        
        name = self._register_point(*origin)
        self._load_commands.append(
            f'\\load{{1}}{{{name}}}[{rotation_deg:.1f}][\\{self.LOAD_SCALE_MACRO_NAME}];'
        )

        if value != '':
            label_x, label_z = self._label_offset_point(
                origin, rotation_deg, self.LOAD_LABEL_DISTANCE
            )
            self._load_commands.append(
                self._fmt_value_node(label_x, label_z, value, anchor='above')
            )
            
    def _add_support(self, style):
        """Erzeugt einen \\support-Befehl von stanli. 
        Wird von jeder Form eines Auflagers aufgerufen, aber dank _support_keys_seen nur einmal ausgeführt."""
        origin = style.get('support_origin')
        support_type = style.get('support_type')
        rotation = style.get('support_rotation', 0)
        if origin is None or support_type is None:
            return

        key = (round(origin[0], 4), round(origin[1], 4))
        if key in self._support_keys_seen:
            return
        self._support_keys_seen.add(key)

        name = self._register_point(*origin)
        self._support_commands.append(f'\\support{{{support_type}}}{{{name}}}[{rotation:.1f}];')
        
    def _add_moment(self, style):
        """Erzeugt einen \\load{2}- bzw. \\load{3}-Befehl für ein Einzelmoment.
        Wird von Bogen und Kopf aufgerufen, durch _moment_keys_seen nur einmal ausgeführt."""
        origin = style.get('load_origin')
        if origin is None:
            return

        key = (round(origin[0], 4), round(origin[1], 4))
        if key in self._moment_keys_seen:
            return
        self._moment_keys_seen.add(key)

        load_type = '2' if style.get('load_clockwise') else '3'
        start_angle, span_angle = (                                 # NEU: Winkel statt Typ-Wechsel
            self.MOMENT_ANGLES_NEGATIVE if style.get('load_clockwise')
            else self.MOMENT_ANGLES_POSITIVE
        )
        value = style.get('load_value', '')
        name = self._register_point(*origin)
        self._load_commands.append(f'\\load{{{load_type}}}{{{name}}}[{start_angle:.1f}][{span_angle:.1f}];')

        if value != '':
            x_str, z_str = self._fmt_scaled_coord_parts(*origin)
            self._load_commands.append(
                self._fmt_value_node(x_str, z_str, value, anchor='above')
            )

    def _add_reaction(self, x, z, style):
        """Wie _add_load, aber für Auflagerkräfte: eigene Befehlsliste und eigenes Dopplungs-Set. 
        So verschlucken sich eine echte Last und eine Auflagerkraft am selben Knoten nicht gegenseitig."""
        origin = style.get('load_origin')
        value = style.get('load_value', '')
        if origin is None:
            return

        key = (round(origin[0], 4), round(origin[1], 4), style.get('load_axis'))
        if key in self._reaction_keys_seen:
            return
        self._reaction_keys_seen.add(key)

        dx = x[0] - x[1]
        dz = z[0] - z[1]
        # siehe Kommentar in _add_load
        angle_deg = np.degrees(np.arctan2(-dz, dx)) + 180
        rotation_deg = (angle_deg + self.LOAD_ANGLE_OFFSET) % 360

        name = self._register_point(*origin)
        self._reaction_commands.append(
            f'\\load{{1}}{{{name}}}[{rotation_deg:.1f}][\\{self.LOAD_SCALE_MACRO_NAME}];'
        )

        if value != '':
            label_x, label_z = self._label_offset_point(
                origin, rotation_deg, self.LOAD_LABEL_DISTANCE
            )
            self._reaction_commands.append(
                self._fmt_value_node(label_x, label_z, value, anchor='above')
            )

    def _add_reaction_moment(self, style):
        """Wie _add_moment, aber für Auflagermomente."""
        origin = style.get('load_origin')
        if origin is None:
            return

        key = (round(origin[0], 4), round(origin[1], 4))
        if key in self._reaction_moment_keys_seen:
            return
        self._reaction_moment_keys_seen.add(key)

        load_type = '2' if style.get('load_clockwise') else '3'
        start_angle, span_angle = (
            self.MOMENT_ANGLES_NEGATIVE if style.get('load_clockwise')
            else self.MOMENT_ANGLES_POSITIVE
        )
        value = style.get('load_value', '')
        name = self._register_point(*origin)
        self._reaction_commands.append(
            f'\\load{{{load_type}}}{{{name}}}[{start_angle:.1f}][{span_angle:.1f}];'
        )

        if value != '':
            x_str, z_str = self._fmt_scaled_coord_parts(*origin)
            self._reaction_commands.append(
                self._fmt_value_node(x_str, z_str, value, anchor='above')
            )

    def _add_hinge(self, style):
        """Erzeugt einen \\hinge-Befehl von stanli. 
        Gleiches Verfahren wie bei _add_support, nur für Gelenke."""
        origin = style.get('hinge_origin')
        hinge_type = style.get('hinge_type')
        if origin is None or hinge_type is None:
            return

        key = (round(origin[0], 4), round(origin[1], 4))    #???
        if key in self._hinge_keys_seen:
            return
        self._hinge_keys_seen.add(key)

        name = self._register_point(*origin)
        self._hinge_commands.append(f'\\hinge{{{hinge_type}}}{{{name}}};')
    
    def add_dimensioning(self, x, z, distance, measure=None, dim_type=None):  # Generiert einen Bemaßungsbefehl mit stanli \dimensioning{type}{start_point}{end_point}{distance}[measure]
        """Erzeugt eine Bemaßung zwischen zwei Punkten mit stanlis \\dimensioning.
        Parameter:
        x, z : je zwei Werte [x0, x1] bzw. [z0, z1] -- Start- und Endpunkt
               der zu bemaßenden Strecke (reale, unskalierte Koordinaten).
        distance : Abstand der Bemaßungslinie von der Strecke. Negativ =
               nach "unten"/"links" verschoben (siehe stanli-Beispiele).
        measure : der angezeigte Text, z.B. '3.0'. Wird automatisch aus der
               Streckenlänge berechnet, falls nicht angegeben.
        dim_type : 1 = horizontale Bemaßung, 2 = vertikale Bemaßung.
               Wird automatisch erkannt (je nachdem ob die Strecke eher
               waagerecht oder senkrecht verläuft), falls nicht angegeben."""
        name_i = self._register_point(x[0], z[0])                           # Punkte wiederverwenden/registrieren,
        name_j = self._register_point(x[1], z[1])                           # genau wie beim Balken

        if dim_type is None:
            dim_type = 1 if abs(x[1] - x[0]) >= abs(z[1] - z[0]) else 2     # erkennt automatisch, ob die Strecke eher horizontal oder vertikal ist

        if measure is None:
            length = np.hypot(x[1] - x[0], z[1] - z[0])
            measure = f'{length:.2f}'
            
        raw_distance = -distance if dim_type == 1 else distance
        distance_str = self._fmt_scaled_value(raw_distance)                    # skaliert den Abstand der Bemaßungslinie mit der TikZ-Variable SCALE_MACRO_NAME
        
        start, end = name_i, name_j                        # Standardmäßig von i nach j
        if dim_type == 2:                                  # für vertikale Bemaßung Richtung umdrehen,
            start, end = name_j, name_i
        
        self._dimensioning_commands.append(
            f'\\dimensioning{{{dim_type}}}{{{start}}}{{{end}}}'
            f'{{{distance_str}}}[${measure}$];'                             # möglicherweise ändern auf [{{\\footnotesize ${measure}$}}]
        )
        
    def _add_dimensioning_from_style(self, style):
        """Liest die von DimensioningGeo mitgegebenen Metadaten aus und reicht sie an die bestehende add_dimensioning()-Methode weiter."""
        point_i = style.get('dim_point_i')
        point_j = style.get('dim_point_j')
        distance = style.get('dim_distance')
        measure = style.get('dim_measure')
        dim_type = style.get('dim_type')
        if point_i is None or point_j is None or distance is None:
            return
        self.add_dimensioning(
            x=[point_i[0], point_j[0]], z=[point_i[1], point_j[1]],
            distance=distance, measure=measure, dim_type=dim_type
        )
      
    def _get_color_name(self, rgba):
        """Registriert eine Farbe einmalig per \\definecolor und gibt deren TikZ-Namen zurück (Wiederverwendung bei gleicher Farbe)."""
        if rgba is None:
            return 'black'                       # Falls keine Farbe übergeben wurde: Standardfall "schwarz"
        key = tuple(round(c, 4) for c in rgba[:3])  # RGB-Werte auf 4 Nachkommastellen runden, um gleiche Farben zu erkennen
        if key not in self._color_names:         # prüft, ob die Farbe bereits registriert wurde
            self._color_counter += 1
            name = f'sscolor{self._color_counter}'
            self._color_names[key] = name
            r, g, b = key                       # entpackt die RGB-Werte in 3 einzelne Variablen
            self._color_defs.append(
                f'\\definecolor{{{name}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}'    # doppelte geschweifte Klammern für LaTeX-Definition
            )
        return self._color_names[key]
    
    def _get_font_size_names(self, size):                           # kompaktes Verfahren wie _get_color_name
        """Registriert eine Schriftgröße einmalig per \\pgfmathsetmacro und gibt die zugehörigen Makronamen zurück 
        (Wiederverwendung bei gleicher Größe, genau wie _get_color_name für Farben)."""
        key = round(size, 2)
        if key not in self._font_size_names:
            letter = self._index_to_letters(len(self._font_size_names))
            fsz_name = f'fsz{letter}'                                   # Namen bestehen bewusst nur aus Buchstaben, keine Ziffern, um TeX-Fehler zu vermeiden
            bsz_name = f'bsz{letter}'
            macro = self.SCALE_MACRO_NAME
            self._font_size_defs.append(
                f'\\pgfmathsetmacro{{\\{fsz_name}}}{{{size * 0.5:.2f}*\\{macro}}}'
            )
            self._font_size_defs.append(
                f'\\pgfmathsetmacro{{\\{bsz_name}}}{{{size * 0.6:.2f}*\\{macro}}}'
            )
            self._font_size_names[key] = (fsz_name, bsz_name)
        return self._font_size_names[key]
    
    def _get_sf_scale_name(self, index, default_value=None):
        """Registriert für jede Schnittkraftfläche (ein Eintrag in StaticForceGeo._segment_data, identifiziert über sf_index) ein eigenes, unabhängiges TikZ-Makro (sfscale1, sfscale2, ...), damit jede Fläche später im .tex-Code individuell nachskaliert werden kann. 
        Alle Makros starten mit demselben Standardwert SF_SCALE_DEFAULT_VALUE. 
        Wiederverwendung bei gleichem Index, genau wie bei _get_color_name/_get_font_size_names."""
        if index not in self._sf_scale_names:
            letter = self._index_to_letters(index)                                    # keine Zahl, sonst funktioniert der Tikz code nicht
            name = f'{self.SF_SCALE_MACRO_PREFIX}{letter}'                           # 1-basiert für Lesbarkeit im .tex
            self._sf_scale_names[index] = name
            value = default_value if default_value is not None else self.SF_SCALE_DEFAULT_VALUE
            self._sf_scale_defs.append(
                f'\\pgfmathsetmacro{{\\{name}}}{{{value}}}'
                f'% <- HIER nur Fläche {index + 1} anpassen (Achtung: Skalierung durch Division)'
            )
        return self._sf_scale_names[index]
    
    def _fmt_calc_coord(self, point_name, angle_deg, along, perp=None):        # NEU
        """Baut eine TikZ-'calc'-Koordinate, die im .tex-Code sichtbar als Summe aus einem bereits registrierten Punkt (z.B. Stabanfang) und einer Strecke entlang/senkrecht zum Stab erscheint.
        So bleibt im erzeugten Code nachvollziehbar, wo genau (z.B. die Extremstelle einer Schnittkraftfläche) relativ zum Stabanfang liegt, statt nur eine bereits fertig verrechnete Zahl zu zeigen.
        TikZ verwendet hier die Polarkoordinaten-Syntax (winkel:radius) innerhalb der 'calc'-Koordinate.
        Parameter
        ---------
        point_name : Name eines bereits registrierten \\point-Punktes (z.B. 'p1')
        angle_deg : Neigungswinkel des Stabes in Grad (0 = horizontal)
        along : bereits mit \\scale skalierte Strecke entlang des Stabes (String)
        perp : bereits mit \\scale skalierte Strecke senkrecht zum Stab (String, optional)
        """
        coord = f'($({point_name})+({angle_deg:.2f}:{along})'
        if perp is not None:
            coord += f'+({angle_deg:.2f}+90:{perp})'
        coord += '$)'
        return coord

    def _add_static_force(self, style):
        """Zeichnet eine Schnittkraftfläche als eine einzige TikZ-Bezier-Kurve (... controls (c1) and (c2) ...), deren Kontrollpunkte über die Hermite-zu-Bezier-Umrechnung exakt aus Wert und Steigung an Stabanfang/-ende berechnet werden. 
        Schraffur läuft über das eingebaute TikZ-Muster 'path picture', angepasst an die exakte Kontur."""
        pivot = style.get('sf_pivot')
        length = style.get('sf_length')
        if pivot is None or length is None:
            return
        
        rotation_deg = style.get('sf_rotation_deg', 0)
        y_i = -style.get('sf_value_i')
        y_j = -style.get('sf_value_j')
        m_i = -style.get('sf_slope_i')
        m_j = -style.get('sf_slope_j')
        
        sf_index = style.get('sf_index')
        sf_macro = (                                                # zusätzliche, individuell einstellbare Skalierung pro Fläche
            self._get_sf_scale_name(sf_index)
            if sf_index is not None else None
        )
        sf_div = f'/\\{sf_macro}' if sf_macro else ''

        stroke_name = self._get_color_name(style.get('stroke_rgba'))
        
        third = length / 3                                      # Bezier-Formel umgestellt um weitere Koordinaten zu erhalten
        c1x, c1y = third, y_i + m_i * third
        c2x, c2y = length - third, y_j - m_j * third

        pivot_name = self._register_point(*pivot)
        macro = self.SCALE_MACRO_NAME
        
        n_hatch = 15                                                        # fester Wert, nach Belieben anpassen
        hatch_step = length / n_hatch if n_hatch else length

        self._static_force_commands.append(
            f'\\begin{{scope}}[shift=({pivot_name}), rotate={rotation_deg:.2f}]'
        )
        self._static_force_commands.append(                                         # Doppelte Variablen, da einmal Anpassung für System und einmal für Skalierung
            f'\\draw[thin, draw={stroke_name}, '
            f'path picture={{'
            f'\\foreach \\h in {{0,{hatch_step:.4f},...,{length:.4f}}}{{'
            f'\\draw [thin] (\\{macro}*\\h,-\\{macro}*3) -- (\\{macro}*\\h,\\{macro}*3);'
            f'}}}}] '
            f'(\\{macro}*0.0000,\\{macro}*0.0000) -- (\\{macro}*0.0000,\\{macro}*{y_i:.4f}{sf_div}) '
            f'.. controls (\\{macro}*{c1x:.4f},\\{macro}*{c1y:.4f}{sf_div}) and (\\{macro}*{c2x:.4f},\\{macro}*{c2y:.4f}{sf_div}) .. '
            f'(\\{macro}*{length:.4f},\\{macro}*{y_j:.4f}{sf_div}) '
            f'-- (\\{macro}*{length:.4f},\\{macro}*0.0000);'
        )
        self._static_force_commands.append('\\end{scope}')
        self._add_static_force_labels(                                    # Beschriftung außerhalb des scope, aber mit Pivot und Rotation
            style, sf_div, stroke_name, pivot_name, rotation_deg
        )
        
    def _add_static_force_labels(self, style, sf_div, stroke_name, pivot_name, rotation_deg):           # NEU (komplette Methode)
        """Schreibt die Wertbeschriftungen (Anfang, Ende, ggf. Extremum) nach der \\scope-Umgebung der dazugehörigen Fläche. 
        Dadurch sitzen sie im TikZ-Code direkt darunter und exakt am selben Punkt wie die Kurve, ein kleiner Versatz kommt nur vom TikZ-Anker (z.B. [below left])"""
        macro = self.SCALE_MACRO_NAME
        labels = style.get('sf_labels') or []
        for (pos, raw_value, text), anchor in zip(labels, self.SF_LABEL_ANCHORS):
            y_val = -raw_value
            along = f'\\{macro}*{pos:.4f}'
            perp = f'\\{macro}*{y_val:.4f}{sf_div}'
            coord = self._fmt_calc_coord(pivot_name, rotation_deg, along, perp)
            self._static_force_commands.append(
                f'\\node[{anchor}, {stroke_name}] at {coord} {{${text}$}};'
            )
            
    def _rotated90_coord_str(self, x, z):
        """Baut den Koordinaten-Inhalt (ohne umschließende Klammern) eines Punktes, der innerhalb einer TikZ-scope mit rotate=90 exakt an seiner wahren Position (x, z) landet, obwohl er dort selbst nochmal um 90 Grad gedreht wird. 
        Ermöglicht es, bestehende \\lineload-Typen (die intern immer nur vertikal versetzen) auch für horizontale Linienlasten (x-Richtung) wiederzuverwenden, ohne dass sich die eigentlichen Endpunkte verschieben.
        Herleitung: 
        ------------
        Die TikZ-Koordinaten sind hier (X,Y) = (x, -z). 
        Eine +90-Grad-Drehung um den Ursprung bildet einen Punkt (a,b) auf (-b,a) ab. 
        Damit der GEDREHTE Punkt am Ende bei (X,Y) landet, muss der UNGEDREHTE ('lokale') Punkt bei (Y,-X) = (-z,-x) liegen."""
        macro = self.SCALE_MACRO_NAME
        local_x = -z
        local_z = -x
        return f'\\{macro}*{local_x:.4f},\\{macro}*{local_z:.4f}'
    
    def _add_lineload_axial(self, style):
        """Zeichnet eine axiale Linienlast (direction='x', coord='bar') als Kette kurzer Pfeile, die in Stabrichtung zeigen. 
        Kein stanli-Befehl verfügbar, daher eigenständig gezeichnet. 
        Der Versatz senkrecht zum Stab funktioniert wie bei den anderen bar-Lasten (Vorzeichen spiegelt die Seite). 
        Die Pfeile zeigen bei positivem Wert von Stabanfang zu Stabende, bei negativem Wert umgekehrt."""
        start = style.get('lineload_start')
        end = style.get('lineload_end')
        val_i = style.get('lineload_value_i', 0.0)
        val_j = style.get('lineload_value_j', 0.0)
        if start is None or end is None:
            return

        x0, z0 = start
        x1, z1 = end
        angle_deg = style.get('lineload_angle_deg', 0.0)
        angle_rad = np.radians(angle_deg)

        sign = 1.0
        if val_i != 0:
            sign = 1.0 if val_i > 0 else -1.0
        elif val_j != 0:
            sign = 1.0 if val_j > 0 else -1.0
        base_mm = sign * self.LINELOAD_DISTANCE_MM

        # Versatz senkrecht zum Stab, direkt in TikZ-Kanvaskoordinaten berechnet
        perp_dx = base_mm * -np.sin(angle_rad)
        perp_dy = base_mm * np.cos(angle_rad)

        scale_macro = self.SCALE_MACRO_NAME
        symbol_macro = self.SYMBOL_SCALE_MACRO_NAME

        def offset_point(x, z):
            base_point = f'\\{scale_macro}*{x:.4f},\\{scale_macro}*{-z:.4f}'
            offset_pos = f'({perp_dx:.4f}mm*\\{symbol_macro},{perp_dy:.4f}mm*\\{symbol_macro})'
            return f'($ ({base_point}) + {offset_pos} $)'

        n_arrows = 5
        direction = 1.0 if sign >= 0 else -1.0
        for k in range(n_arrows):
            t0 = k / n_arrows
            t1 = (k + 0.7) / n_arrows      # kurze Pfeile mit kleiner Lücke dazwischen
            if direction < 0:
                t0, t1 = 1 - t0, 1 - t1
            xa, za = x0 + (x1 - x0) * t0, z0 + (z1 - z0) * t0
            xb, zb = x0 + (x1 - x0) * t1, z0 + (z1 - z0) * t1
            self._lineload_commands.append(
                f'\\draw[->] {offset_point(xa, za)} -- {offset_point(xb, zb)};'
            )

        self._add_lineload_labels(style, start, end, sign, extra_rotation_deg=0.0)
    
    def _add_lineload_far_labels(self, labels, coord, z0, z1, val_i, val_j, base_mm):
        """Positioniert die Beschriftung(en) einer eigenständig gezeichneten Linienlast mittig auf der ÄUSSEREN (fernen) Linie, 
        mit kleinem zusätzlichem Abstand darüber hinaus. Nutzt dieselbe coord()-Funktion wie die Last selbst (als Parameter 
        übergeben), damit Beschriftung und Linie garantiert konsistent zueinander liegen -- unabhängig davon, wie x_ref, 
        Vorzeichen oder Stabrichtung im Einzelfall berechnet wurden.
        base_mm bestimmt die Seite: negativ = Last liegt links, Beschriftung wird noch etwas weiter nach links geschoben (und
        umgekehrt), damit sie nicht auf der Linie, sondern klar daneben sitzt."""
        if not labels:
            return

        anchor = 'left' if base_mm >= 0 else 'right'
        clearance_sign = 1 if base_mm >= 0 else -1
        xshift = f'{clearance_sign * self.LINELOAD_LABEL_CLEARANCE_MM:.2f}mm'

        fsz_name, bsz_name = self._get_font_size_names(self._load_label_font_pt)
        font = f'font=\\fontsize{{\\{fsz_name} pt}}{{\\{bsz_name} pt}}\\selectfont'

        if len(labels) == 1:
            points = [((z0 + z1) / 2, (val_i + val_j) / 2, labels[0])]
        else:
            points = [(z0, val_i, labels[0]), (z1, val_j, labels[1])]

        for z, val, value in points:
            if value in ('', None):
                continue
            point = coord(z, val)
            self._lineload_commands.append(
                f'\\node[{anchor}, xshift={xshift}, {font}] at {point} {{${value}$}};'
            )
    
    def _add_lineload_x_proj(self, style):
        """Zeichnet eine Linienlast mit direction='x', coord='system', length='proj' eigenständig mit \\draw-Befehlen. 
        Stanli bietet dafür keinen passenden \\lineload-Typ (siehe vorheriger Kommentar). 
        Verankerung: bei positivem Wert am Stabanfang (node_i), bei negativem Wert am Stabende (node_j). 
        Dafür werden die originalen x-Werte des Stabes benötigt (lineload_bar_x_i/j), da _handle_x_direction 
        in effect.py beide x-Werte sonst auf denselben Wert zusammenzieht."""
        start = style.get('lineload_start')
        end = style.get('lineload_end')
        val_i = style.get('lineload_value_i', 0.0)
        val_j = style.get('lineload_value_j', 0.0)
        if start is None or end is None:
            return

        _, z0 = start
        _, z1 = end

        sign = 1.0
        if val_i != 0:
            sign = 1.0 if val_i > 0 else -1.0
        elif val_j != 0:
            sign = 1.0 if val_j > 0 else -1.0
        base_mm = -sign * self.LINELOAD_DISTANCE_MM

        bar_x_i = style.get('lineload_bar_x_i')
        bar_x_j = style.get('lineload_bar_x_j')
        if bar_x_i is not None and bar_x_j is not None:
            x_ref = bar_x_i if sign >= 0 else bar_x_j
        else:
            x_ref = start[0]     # Rueckfallwert, falls die Werte einmal fehlen sollten

        scale_macro = self.SCALE_MACRO_NAME
        symbol_macro = self.SYMBOL_SCALE_MACRO_NAME
        load_macro = self.LOAD_SCALE_MACRO_NAME

        def coord(z, extra_val=0.0):
            z_str = f'\\{scale_macro}*{-z:.4f}'
            base_point = f'(\\{scale_macro}*{x_ref:.4f},{z_str})'
            offset_point = f'({base_mm:.4f}mm*\\{symbol_macro},0)'
            arrow_val = -sign * abs(extra_val)
            arrow_point = f'({arrow_val:.4f}*\\{load_macro} cm,0)'
            return f'($ {base_point} + {offset_point} + {arrow_point} $)'

        near0, near1 = coord(z0), coord(z1)
        far0, far1 = coord(z0, val_i), coord(z1, val_j)

        self._lineload_commands.append(f'\\draw[thin] {near0} -- {near1};')
        self._lineload_commands.append(f'\\draw[thin] {far0} -- {far1};')
        self._lineload_commands.append(f'\\draw[thin] {near0} -- {far0};')
        self._lineload_commands.append(f'\\draw[thin] {near1} -- {far1};')

        n_arrows = 6
        for k in range(n_arrows + 1):
            t = k / n_arrows
            z_t = z0 + (z1 - z0) * t
            val_t = val_i + (val_j - val_i) * t
            if abs(val_t) < 1e-6:
                continue
            # von nah nach fern: Pfeilspitze zeigt in Richtung des Vorzeichens von val_t
            self._lineload_commands.append(f'\\draw[->] {coord(z_t, val_t)} -- {coord(z_t)};')

        self._add_lineload_far_labels(
            style.get('lineload_labels'), coord, z0, z1, val_i, val_j, base_mm
        )        
    def _add_lineload(self, style):
        """Fasst alle Einzelteile einer Linienlast, die LineArrowGeo normalerweise  erzeugt (mehrere Pfeile + Begrenzungslinien) zu einem \\lineload-Befehl zusammen. 
        Dank _lineload_keys_seen in __init__ wird nur beim ersten der vielen Aufrufe tatsächlich etwas gezeichnet, alle weiteren (die einzelnen Pfeile) werden stillschweigend verworfen.
        Für direction='x'+coord='system'+length='exact' (stanli-Typ 2) wird der komplette Aufruf zusätzlich in eine um 90 Grad gedrehte TikZ-scope verpackt, damit die \\lineload-Logik auch horizontale Lasten korrekt zeichnet."""
        start = style.get('lineload_start')
        end = style.get('lineload_end')
        if start is None or end is None:
            return

        key = (
            round(start[0], 4), round(start[1], 4),
            round(end[0], 4), round(end[1], 4),
        )
        if key in self._lineload_keys_seen:
            return
        self._lineload_keys_seen.add(key)

        lineload_type = style.get('lineload_type', 2)
        if lineload_type == 'x_proj':
            self._add_lineload_x_proj(style)
            return
        if lineload_type == 'x_axial':                # NEU
            self._add_lineload_axial(style)
            return
        axis = style.get('lineload_axis', 'z')
        # Nur Typ 2 (direction='x', coord='system', length='exact') lässt sich per Drehung korrekt wiederverwenden - Typ 1 (bar) und Typ 3 (proj) nicht.
        rotate_for_x = lineload_type == 2 and axis == 'x'

        if rotate_for_x:
            # eigene, nicht über _register_point zwischengespeicherte Koordinaten, da sie nur innerhalb dieser gedrehten scope die richtige Position ergeben
            coord_i = self._rotated90_coord_str(*start)
            coord_j = self._rotated90_coord_str(*end)
        else:
            coord_i = self._register_point(*start)
            coord_j = self._register_point(*end)

        # lineload_value_i/j sind auf -1..1 normiert (Anteil am größten Lastwert im System).
        # Durch \loadscale wird die Last wie die Einzelkräfte skaliert, um Pfeillänge zu beeinflussen
        # Ein Wert von 0 lässt stanli den Pfeil an diesem Ende automatisch weg (das ist genau der Fall für Dreieckslasten).
        macro = self.LOAD_SCALE_MACRO_NAME
        val_i = style.get('lineload_value_i', 0.0)
        val_j = style.get('lineload_value_j', 0.0)
        len_i = f'{val_i:.4f}*\\{macro}'
        len_j = f'{val_j:.4f}*\\{macro}'
        
        # Stabseite: Positiver Wert -> Standardseite, negativer Wert -> gespiegelt.
        # Wert links (val_i) entscheidet die Seite, außer es ist 0, dann entscheidet der rechte Wert (val_j).
        sign = 1.0
        if val_i != 0:
            sign = 1.0 if val_i > 0 else -1.0
        elif val_j != 0:
            sign = 1.0 if val_j > 0 else -1.0
            
        # \lineloadDistance ist ein fester Abstand in structuranalysis.sty, welcher hierbei nur innerhalb der folgenden {}-Gruppe mit passendem Vorzeichen überschrieben wird.
        # Anschließend gilt wieder der ursprüngliche Wert.
        distance_value = f'{sign * self.LINELOAD_DISTANCE_MM:.4f}mm*\\{self.SYMBOL_SCALE_MACRO_NAME}'


        if lineload_type == 3:
            distance = style.get('lineload_distance', 0.0)
            distance_str = self._fmt_scaled_value(distance)
            lineload_cmd = (
                f'\\lineload{{3}}{{{coord_i}}}{{{coord_j}}}'
                f'[{len_i}][{len_j}][{distance_str}];'
            )
        else:
            lineload_cmd = (
                f'\\lineload{{{lineload_type}}}{{{coord_i}}}{{{coord_j}}}'
                f'[{len_i}][{len_j}];'
            )

        if rotate_for_x:
            self._lineload_commands.append('\\begin{scope}[rotate=90]')
        else:
            self._lineload_commands.append('{')
        self._lineload_commands.append(
            f'\\renewcommand{{\\lineloadDistance}}{{{distance_value}}}'
        )
        self._lineload_commands.append(lineload_cmd)
        self._lineload_commands.append(
            '\\end{scope}' if rotate_for_x else '}'
        )
        
        extra_rotation = 90.0 if rotate_for_x else 0.0
        self._add_lineload_labels(style, start, end, sign, extra_rotation) 

    def _add_lineload_labels(self, style, start, end, sign, extra_rotation_deg=0.0): 
        """Schreibt die Wertbeschriftung(en) einer Linienlast in festem, von der Systemgröße unabhängigem Abstand neben die \\lineload- Linie. 
        Wird direkt aus _add_lineload heraus aufgerufen, wodurch sind Last und Beschriftung immer zusammen sichtbar oder ausgeblendet (statt getrennt über add_text).
        'sign' spiegelt die Beschriftung auf dieselbe Seite wie die (ggf. gespiegelte) Last, 'extra_rotation_deg' dreht die Versatzrichtung zusätzlich mit, falls die Last selbst für die x-Richtung gedreht gezeichnet wurde."""
        labels = style.get('lineload_labels')
        if not labels:
            return

        angle_deg = style.get('lineload_angle_deg', 0.0) + extra_rotation_deg
        angle_rad = np.radians(angle_deg)
        side = self.LINELOAD_LABEL_SIDE * sign
        disp_x = side * self.LINELOAD_LABEL_DISTANCE * -np.sin(angle_rad)
        disp_z = side * self.LINELOAD_LABEL_DISTANCE * np.cos(angle_rad)

        x0, z0 = start
        x1, z1 = end
        if len(labels) == 1:
            points = [((x0 + x1) / 2, (z0 + z1) / 2, labels[0])]
        else:
            points = [(x0, z0, labels[0]), (x1, z1, labels[1])]

        for x, z, value in points:
            if value in ('', None):
                continue
            label_x, label_z = self._scaled_offset_coord(x, z, disp_x, disp_z)
            self._lineload_commands.append(
                self._fmt_value_node(label_x, label_z, value, anchor='above')
            )
            
    @staticmethod
    def _index_to_letters(index):
        """Wandelt 0, 1, 2, ... in a, b, c, ... um, damit die Namen auch bei sehr vielen unterschiedlichen Schriftgrößen garantiert ziffernfrei bleiben."""
        letters = ''
        index += 1
        while index > 0:
            index, rem = divmod(index - 1, 26)
            letters = chr(ord('a') + rem) + letters
        return letters

    def add_graphic(self, x, z, **style):
        """Zeichnet eine Linie oder ein Polygon als \draw-Befehl in TikZ."""
        if x is None or z is None or len(x) == 0:   # Abbruchbedingung, falls keine Koordinaten übergeben wurden
            return
        if style.get('element_type') == 'suppress_tikz':   # wird komplett verworfen/unterdrückt
            return
        
        if style.get('element_type') == 'hinge':
            self._add_hinge(style)
            return
    
        if style.get('element_type') == 'support':  
            self._add_support(style)      
            return
    
        if style.get('element_type') == 'beam' and len(x) == 2:    
            self._add_beam(x, z)                                   
            return                                                 # hier abbrechen, kein \draw erzeugen

        if style.get('element_type') == 'load':                    # Einzellasten und Momente
            if not self._show_loads:                               # Lasten komplett unterdrücken, falls deaktiviert
                return
            if style.get('load_kind') == 'moment':
                self._add_moment(style)
                return
            if len(x) == 2:
                self._add_load(x, z, style)
            return
        
        if style.get('element_type') == 'reaction':                # Auflagerkräfte (Kräfte und Moment)
            if not self._show_reactions:                           # unterdrücke die Darstellung der Auflagerkräfte, wenn die Option deaktiviert ist
                return
            if style.get('load_kind') == 'moment':
                self._add_reaction_moment(style)
                return
            if len(x) == 2:
                self._add_reaction(x, z, style)
            return

        if style.get('element_type') == 'static_force':
            self._add_static_force(style)
            return 
        
        if style.get('element_type') == 'lineload':
            if not self._show_loads:                                # Linienlasten unterdrücken, wenn show_loads deaktiviert ist
                return
            self._add_lineload(style)                               # bla bla bla
            return

        
        if style.get('element_type') == 'dimensioning' and len(x) == 2:
            if not self._show_dimensioning:                                 # Wenn Bemaßung ausgaschaltet hier abbrechen
                return
            self._add_dimensioning_from_style(style)
            return 
        
        coords = ' -- '.join(                       # verbindet die Koordinatenpaare zu einem TikZ-Pfad
            self._fmt_coord(xi, zi) for xi, zi in zip(x, z)
            if xi is not None and zi is not None
        )
        if not coords or ' -- ' not in coords:      # Abbruch bei unvollständigen Koordinaten
            return

        options = []                                # Liste der TikZ-Optionen für den \draw-Befehl
        is_fill = style.get('fill', False)

        if is_fill:
            fill_name = self._get_color_name(style.get('fill_rgba'))
            options.append(f'fill={fill_name}')

        stroke_rgba = style.get('stroke_rgba')
        if stroke_rgba is not None or not is_fill:
            stroke_name = self._get_color_name(stroke_rgba)
            options.append(f'draw={stroke_name}')

        if 'linewidth_pt' in style:
            options.append(f'line width={style["linewidth_pt"]:.2f}pt')

        if style.get('dash', 'solid') != 'solid':
            options.append(style['dash'])

        if 'opacity' in style:
            options.append(f'opacity={style["opacity"]:.2f}')

        opts_str = ', '.join(options)               # alle TikZ-Optionen zu einem String zusammenfügen
        suffix = ' -- cycle' if is_fill else ''     # schließt das Polygon, falls es gefüllt ist
        self._draw_commands.append(f'\\draw[{opts_str}] {coords}{suffix};')
        
    @staticmethod
    def _foreach_range(start, end, step):
        """Erzeugt eine Liste von Werten von start bis end mit dem angegebenen Schritt.
        Wird für die Achsenbeschriftung und das Gitter benötigt."""
        second = start + step
        return f'{start:g},{second:g},...,{end:g}'

    def _axis_commands(self):
        """Erzeugt TikZ-Code für Koordinatenachsen (show_axis) und/oder ein
        Hintergrundgitter (show_grid) -- jetzt über TikZ' eigene \\foreach-
        Schleife statt einzeln generierter Zeilen pro Tick."""
        if not (self._show_axis or self._show_grid):
            return []

        step = self.AXIS_TICK_STEP
        x_min, x_max, z_min, z_max = self.scene_boundaries
        margin = step * self.AXIS_MARGIN_TICKS

        x_start = np.floor((x_min - margin) / step) * step
        x_end = np.ceil((x_max + margin) / step) * step
        z_start = np.floor((z_min - margin) / step) * step
        z_end = np.ceil((z_max + margin) / step) * step

        x_range = self._foreach_range(x_start, x_end, step)
        z_range = self._foreach_range(z_start, z_end, step)
        macro = self.SCALE_MACRO_NAME

        commands = []

        if self._show_grid:
            z_start_x, z_start_z = self._fmt_scaled_coord_parts(0, z_start)  # nur zum Klammern-Muster wiederverwenden
            z_end_x, z_end_z = self._fmt_scaled_coord_parts(0, z_end)
            x_start_x, x_start_z = self._fmt_scaled_coord_parts(x_start, 0)
            x_end_x, x_end_z = self._fmt_scaled_coord_parts(x_end, 0)

            commands.append('% --- Hintergrundgitter (per \\foreach) ---')
            commands.append(
                f'\\foreach \\gx in {{{x_range}}}{{'
                f'\\draw[gray!25, thin] (\\{macro}*\\gx,{z_start_z}) -- '
                f'(\\{macro}*\\gx,{z_end_z});}}'
            )
            commands.append(
                f'\\foreach \\gz in {{{z_range}}}{{'
                f'\\draw[gray!25, thin] ({x_start_x},\\{macro}*-\\gz) -- '
                f'({x_end_x},\\{macro}*-\\gz);}}'
            )

        if self._show_axis:
            axis_z = z_end
            axis_x = x_start
            axis_z_str_pos = self._fmt_scaled_coord_parts(0, axis_z - 0.05)[1]
            axis_z_str_neg = self._fmt_scaled_coord_parts(0, axis_z + 0.05)[1]
            axis_z_label_str = self._fmt_scaled_coord_parts(0, axis_z + 0.18)[1]
            axis_x_str = self._fmt_scaled_coord_parts(axis_x, 0)[0]

            x_axis_start = self._fmt_coord(x_start, axis_z)
            x_axis_end = self._fmt_coord(x_end, axis_z)

            commands.append('% --- x-Achse am Rand (per \\foreach) ---')
            commands.append(
                f'\\draw[->, thin] {x_axis_start} -- {x_axis_end} node[right] {{$x$}};'
            )
            commands.append(
                f'\\foreach \\tx in {{{x_range}}}{{'
                f'\\draw[thin] (\\{macro}*\\tx,{axis_z_str_pos}) -- '
                f'(\\{macro}*\\tx,{axis_z_str_neg});'
                f'\\node[below, font=\\tiny] at (\\{macro}*\\tx,{axis_z_label_str}) {{$\\tx$}};}}'
            )

            axis_x_str_pos = self._fmt_scaled_coord_parts(axis_x - 0.05, 0)[0]
            axis_x_str_neg = self._fmt_scaled_coord_parts(axis_x + 0.05, 0)[0]
            axis_x_label_str = self._fmt_scaled_coord_parts(axis_x - 0.18, 0)[0]

            z_axis_start = self._fmt_coord(axis_x, z_start)
            z_axis_end = self._fmt_coord(axis_x, z_end)

            commands.append('% --- z-Achse am Rand (per \\foreach) ---')
            commands.append(
                f'\\draw[->, thin] {z_axis_start} -- {z_axis_end} node[below] {{$z$}};'
            )
            commands.append(
                f'\\foreach \\tz in {{{z_range}}}{{'
                f'\\draw[thin] ({axis_x_str_pos},\\{macro}*-\\tz) -- '
                f'({axis_x_str_neg},\\{macro}*-\\tz);'
                f'\\node[left, font=\\tiny] at ({axis_x_label_str},\\{macro}*-\\tz) {{$\\tz$}};}}'
            )
        return commands
    
    @staticmethod
    def _number_to_letter(text_str):
        """Konvertiert eine Zahl als String ('1', '2', '3', ...) in den entsprechenden Kleinbuchstaben ('a', 'b', 'c', ...)."""
        if text_str.isdigit():
            index = int(text_str) - 1           # berechnet den Index basierend auf der Zahl (1 -> 0, 2 -> 1, ...)
            if 0 <= index < 26:                  # prüft, ob der Index im Bereich der Buchstaben a-z liegt
                return chr(ord('a') + index)    # gibt den entsprechenden Buchstaben zurück (chr(ord('a') + index) berechnet den Buchstaben basierend auf dem ASCII-Wert von 'a')
        return text_str

    def _coord_or_point(self, x, z):
        """Prüft, ob an dieser Stelle bereits ein \\point-Punkt registriert wurde.
        Falls ja, wird dessen Name zurückgegeben (z.B. '(p3)'), was TikZ genauso interpretiert wie eine Koordinate. 
        Ansonsten wird wie bisher die skalierte Koordinate ausgegeben."""
        key = (round(x, 4), round(z, 4))
        if key in self._point_names:
            return f'({self._point_names[key]})'
        return self._fmt_coord(x, z)
    
    def add_text(self, x, z, text, rotation=0, label_type=None, **style):
        """Fügt einen Textknoten in TikZ hinzu, mit optionaler Rotation und Stil. 
        Je nach label_type wird der Text als als Auflagerbezeichnung, Stabnummer oder allgemeiner Text behandelt."""
        if style.get('element_type') in (
            'load', 'reaction', 'dimensioning', 'static_force', 'lineload'
        ):              # Lastenbeschriftungen werden bereits in _add_load behandelt, daher hier überspringen
            return
        if label_type == 'support' and not self._show_support_labels:
            return
        if label_type == 'bar_number' and not self._show_bar_numbers:
            return
            
        text_str = (
            ' '.join(str(t) for t in text) if isinstance(text, (list, tuple)) else str(text)    # prüft, ob der Text eine Liste oder ein Tupel ist, und verbindet die Elemente zu einem String, ansonsten wird der Text direkt in einen String umgewandelt
        )
        if label_type == 'support':
            text_str = TikzRenderer._number_to_letter(text_str)     # konvertiert bei einem Auflager die Zahl in den entsprechenden Buchstaben (1 -> a, 2 -> b, ...)
            
        # LaTeX-Sonderzeichen erstellen, damit der Code fehlerfrei kompiliert werden kann
        for char, escaped in (('_', '\\_'), ('%', '\\%'), ('&', '\\&')):
            text_str = text_str.replace(char, escaped)
            
        node_options = []                             # Optionen, die am einzelnen Node bleiben (nicht in \scope-Umgebung)
        if rotation:
            node_options.append(f'rotate={rotation:.1f}')
            
        anchor = style.get('anchor')                       # generischer Anker für Positionierung
        if anchor:
            node_options.append(anchor)
        elif label_type == 'support':                        # above für Knotenbezeichnungen
            node_options.append('above')
            
        if label_type == 'bar_number':
            node_options.append('above')
            node_options.append('draw')
            node_options.append('circle')
            node_options.append('inner sep=1pt')
            node_options.append('line width=0.3pt')
            
        color_name = (                                 # Farbe wird in \scope-Umgebung gesetzt
            self._get_color_name(style['text_rgba']) if 'text_rgba' in style
            else None
        )
        font_key = (                                   # Schriftgröße genauso
            self._get_font_size_names(style['font_size_pt'])
            if 'font_size_pt' in style else None
        )

        coord = self._coord_or_point(x, z)
        
        node_text = (           
            f'{{${text_str}$}}' if label_type == 'support'        # Auflagerbezeichnungen kursiv darstellen (Mathemodus) unabhängig von der Schriftgröße  
            else text_str
        )
        
        self._text_node_specs.append({
            'font_key': font_key,
            'color_name': color_name,
            'node_options': node_options, 
            'coord': coord,
            'node_text': node_text,
        })
    def _build_text_commands(self):
        """Gruppiert alle gesammelten Textknoten nach (Schriftgröße, Farbe) und baut daraus \\begin{scope}[...]-Blöcke.
        Schriftart/Farbe werden nur einmal definiert, anstatt an jedem einzelnen \\node wiederholt zu werden."""
        groups = {}
        order = []
        for spec in self._text_node_specs:
            key = (spec['font_key'], spec['color_name'])
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(spec)

        commands = []
        for key in order:
            font_key, color_name = key
            scope_opts = []
            if font_key:
                fsz_name, bsz_name = font_key
                scope_opts.append(
                    f'font=\\fontsize{{\\{fsz_name} pt}}{{\\{bsz_name} pt}}\\selectfont'
                )
            if color_name:
                scope_opts.append(f'text={color_name}')
            opts_str = ', '.join(scope_opts)

            commands.append(f'\\begin{{scope}}[{opts_str}]')
            for spec in groups[key]:
                node_opts_str = ', '.join(spec['node_options'])
                commands.append(
                    f'\\node[{node_opts_str}] at {spec["coord"]} '
                    f'{{{spec["node_text"]}}};'
                )
            commands.append('\\end{scope}')
        return commands
    
    def _wrap_reaction_scope(self):
        """Verpackt alle Auflagerkraft-Befehle in eine rote TikZ-scope-Umgebung. 
        So unterscheiden sie sich optisch von echten Lasten (schwarz), ohne dass \\load selbst eine Farboption bräuchte, da die Befehle in \\load die Farbe einfach von der umgebenden scope erben."""
        if not self._reaction_commands:
            return []
        return (
            ['\\begin{scope}[red]']
            + self._reaction_commands
            + ['\\end{scope}']
        )
        
    @staticmethod
    def _section_header(title):
        """Baut einen optisch abgesetzten Kommentarblock als Abschnittsüberschrift im generierten .tex-Code, für bessere Übersicht bei längeren, komplexeren Systemen."""
        bar = '%' * (len(title) + 8)
        return [bar, f'%%% {title} %%%', bar]

    @cached_property                # cached_property sorgt dafür, dass die Methode nur einmal ausgeführt wird und das Ergebnis zwischengespeichert wird
    def figure(self):
        """Baut aus allen gesammelten Farbdefinitionen und Zeichenbefehlen den fertigen TikZ-Code."""
        scale_line = (
            f'\\pgfmathsetmacro{{\\{self.SCALE_MACRO_NAME}}}'       # definiert die Skalierungsvariable in TikZ für spätere Anpassung in der LaTeX-Datei
            f'{{{self.SCALE_DEFAULT_VALUE}}}  '
            f'% <- HIER Skalierung anpassen (1.0 = 100%)'
        )
        load_scale_line = (                                             
            f'\\pgfmathsetmacro{{\\{self.LOAD_SCALE_MACRO_NAME}}}'
            f'{{{self.LOAD_SCALE_DEFAULT_VALUE}}}  '
            f'% <- HIER NUR die Pfeillänge der Lasten anpassen (unabhängig von \\scale)'
        )
        symbol_scale_line = (                                        # Skalierung von Symbolen
        f'\\pgfmathsetmacro{{\\{self.SYMBOL_SCALE_MACRO_NAME}}}'
        f'{{{self.SYMBOL_SCALE_DEFAULT_VALUE}}}  '
        f'% <- HIER NUR die Größe von Auflagern/Gelenken anpassen (unabhängig von \\scale)'
        )
        body = '\n'.join(                                                                                       # Reihenfolge ist wichtig!
            [scale_line, load_scale_line, symbol_scale_line, ''] 
            + self._section_header('Schnittkraftflächen-Skalierung') + self._sf_scale_defs + ['']
            + self._section_header('Symbolgrößen') + self._symbol_scale_overrides() + ['']
            + self._section_header('Farben & Schriftgrößen') + self._color_defs+ self._font_size_defs + ['']
            + self._section_header('Achsen & Gitter') + self._axis_commands() + ['']                            # erst die Achsen und das Gitter, falls aktiviert ...
            + self._section_header('Punkte') + self._point_commands + ['']                                      # ... dann alle \point-Befehle ...
            + self._section_header('Auflager') + self._support_commands + ['']                                  # ... dann alle \support-Befehle ...    
            + self._section_header('Stäbe') + self._beam_commands + ['']                                        # ... dann alle \beam-Befehle ...
            + self._section_header('Gelenke') + self._hinge_commands + ['']                                     # ... dann alle \hinge-Befehle ...
            + self._section_header('Lasten') + self._load_commands + ['']                                       # ... dann alle \load-Befehle ...
            + self._section_header('Linienlasten') + self._lineload_commands + ['']                             # ... dann alle \lineload-Befehle ...
            + self._section_header('Bemaßung') + self._dimensioning_commands + ['']                             # ... dann alle \dimensioning-Befehle ...
            + self._section_header('Schnittkraftflächen') + self._static_force_commands + ['']
            + self._section_header('Auflagerkräfte') + self._wrap_reaction_scope() + ['']
            + self._section_header('Zeichenbefehle') + self._draw_commands + ['']                               # ... dann der Rest (\draw, \node)
            + self._section_header('Beschriftungen') + self._build_text_commands()                              # ... für die scope-Blöcke der Textknoten
        )
        # Aufbau der TikZ-Umgebung mit \begin{tikzpicture} und \end{tikzpicture}
        return (
            '\\begin{tikzpicture}[>=latex]\n'
            f'{body}\n'
            '\\end{tikzpicture}'
        )

    def show(self):
        """Zeigt je nach Einstellung den TikZ-Code, das PDF, etc."""
        if self._show_code:             # zeigt den TikZ-Code im Terminal an, falls self._show_code True ist
            print(self.figure)
 
        if self._show_pdf:              # rendert das PDF und öffnet es, falls self._show_pdf True ist
            self._render_pdf()
            
        if self._save_tex:              # speichert die .tex-Datei, falls self._save_tex True ist
            self.save_tex_file(self._tex_filename)

        if self._save_tikz:             # speichert die reine .tikz-Datei, falls self._save_tikz True ist
            self.save_tikz_file(self._tikz_filename)
        
    def _build_document(self):
        """Verpackt das reine tikzpicture in ein vollständiges, eigenständig kompilierbares LaTeX-Dokument."""
        return (
            '\\documentclass[tikz,border=10pt]{standalone}\n'
            '\\usepackage{tikz}\n'
            '\\usetikzlibrary{calc}\n'                          # ermöglicht ($...$)-Rechnungen in Koordinaten
            '\\usepackage{structuralanalysis}\n'
            '\\usepackage{3dstructuralanalysis}\n'
            '\\begin{document}\n'
            f'{self.figure}\n'
            '\\end{document}\n'
            )

    @staticmethod
    def _next_available_filename(out_dir, base='sstatics_output', ext='.tex'):
        """Sucht im Zielordner den nächsten freien, durchnummerierten Dateinamen, z.B. sstatics_output_1.tex, sstatics_output_2.tex, ...
        Bestehende Dateien werden dabei nicht überschrieben."""
        import os

        counter = 1
        while True:
            candidate = f'{base}_{counter}{ext}'
            if not os.path.exists(os.path.join(out_dir, candidate)):
                return candidate
            counter += 1
            
    def save_tex_file(self, filename: str | None = None):
        """Speichert den vollständigen, kompilierbaren LaTeX-Code (siehe _build_document) als .tex-Datei, standardmäßig im Downloads-Ordner des Benutzers.
        Parameter
        ---------
        filename : str | None
            Dateiname der .tex-Datei. Falls None, wird automatisch 'sstatics_output_1.tex' (oder die nächste verfügbare Nummer) verwendet.
        Returns
        ---------
        str
            Der vollständige Pfad der gespeicherten Datei.
        """
        import os
        from pathlib import Path        # für einen plattformunabhängigen Pfad zum Downloads-Ordner
    

        if self._tex_output_dir is not None:
            out_dir = self._tex_output_dir
        else:
            out_dir = str(Path.home() / 'Downloads')

        os.makedirs(out_dir, exist_ok=True)         # legt einen Downloads-Ordner an, falls er nicht existiert und exist_ok=True sorgt dafür, dass kein Fehler auftritt, falls er bereits existiert

        if filename is None:
            filename = self._next_available_filename(out_dir)
        elif not filename.endswith('.tex'):
            filename += '.tex'
            
        out_path = os.path.join(out_dir, filename)

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self._build_document())

        print(f'TeX-Datei gespeichert unter: {out_path}')
        return out_path
    
    def save_tikz_file(self, filename: str | None = None):
        """Speichert nur das reine tikzpicture als .tikz-Datei, standardmäßig im Downloads-Ordner des Benutzers. 
        Wird z.B. für die Einbindungen per \\include{} in ein eigenes Dokument benötigt.
        Parameter
        ---------
        filename : str | None
            Dateiname der .tikz-Datei. Falls None, wird automatisch 'sstatics_output.tikz' verwendet.
        Returns
        ---------
        str
            Der vollständige Pfad der gespeicherten Datei.
        """
        import os
        from pathlib import Path

        if self._tex_output_dir is not None:
            out_dir = self._tex_output_dir
        else:
            out_dir = str(Path.home() / 'Downloads')

        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            filename = self._next_available_filename(out_dir, ext='.tikz')
        elif not filename.endswith('.tikz'):
            filename += '.tikz'

        out_path = os.path.join(out_dir, filename)

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.figure)

        print(f'TikZ-Datei (nur tikzpicture) gespeichert unter: {out_path}')
        return out_path
    
    def _render_pdf(self):
        """Kompiliert den TikZ-Code zu PDF und öffnet es (wie plt.show())."""
        import os
        import subprocess
        import sys
        import tempfile

        out_dir = os.path.join(tempfile.gettempdir(), 'sstatics_tikz_preview')
        os.makedirs(out_dir, exist_ok=True)
        tex_path = os.path.join(out_dir, 'preview.tex')

        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(self._build_document())

        try:
            subprocess.run(
                [
                    'pdflatex',
                    '-interaction=nonstopmode',
                    '-output-directory', out_dir,
                    tex_path
                ],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            print('Fehler beim Kompilieren des TikZ-Codes:')
            print(e.stdout.decode(errors='ignore')[-2000:])
            return
        except FileNotFoundError:
            print(
                'pdflatex wurde nicht gefunden. Prüfe, ob MiKTeX korrekt installiert und im PATH ist.'
            )
            return

        pdf_path = os.path.join(out_dir, 'preview.pdf')
        if sys.platform == 'win32':
            os.startfile(pdf_path)
        else:
            opener = 'open' if sys.platform == 'darwin' else 'xdg-open'     # für macOS oder Linux
            subprocess.run([opener, pdf_path])