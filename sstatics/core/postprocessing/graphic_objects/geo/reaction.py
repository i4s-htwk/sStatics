from sstatics.core.preprocessing.loads import NodePointLoad
from sstatics.core.postprocessing.graphic_objects.geo.effect import (
    PointLoadGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.system import SystemGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_STATE_LINE, DEFAULT_STATE_LINE_TEXT
)


class ReactionForceGeo(PointLoadGeo):
    """
    Baut aus einem gelösten Solver (z.B. FirstOrder) die grafischen Objekte für die Auflagerkräfte (Reaktionskräfte).

    Für jeden Knoten, der ein Auflager (oder eine Feder) besitzt, wird ein PointLoadGeo erzeugt (dieselbe Klasse, die auch für normale Einzellasten an Knoten verwendet wird). Der einzige
    Unterschied ist die Markierung 'element_type': 'reaction' statt 'load'. 
    Dadurch zeichnet der TikzRenderer die Auflagerkräfte in einer eigenen Gruppe und verwechselt sie nicht mit echten, aufgebrachten Lasten am selben Knoten.
    """

    @staticmethod
    def _is_supported(node):
        """Ein Knoten hat ein Auflager, wenn mindestens eine seiner drei Freiheitsgrade (u, w, phi) NICHT 'free' ist. 
        Das deckt sowohl starre Lager ('fixed') als auch Federlager (Zahlenwert als Federsteifigkeit) ab."""
        return any(
            getattr(node, attr) != 'free' for attr in ('u', 'w', 'phi')
        )

    @classmethod
    def elements_from_solver(
            cls,
            solver,
            global_scale: float,
            decimals: int = 2,
            sig_digits: int | None = None,
            distance: float | None = None,
            show_text: bool = True,
    ):
        """Erzeugt die Systemgeometrie (SystemGeo) und eine Liste von PointLoadGeo-Objekten für die Auflagerkräfte.
        Parameter:
        solver : Solver
            Eine bereits gelöste Solver-Instanz (z.B. FirstOrder), aus der über 'system_support_forces' die Auflagerreaktionen im globalen Koordinatensystem gelesen werden.
        bar_mesh_type : str, default='bars'
            Welche Netz-Variante für die Systemgeometrie verwendet wird.
        decimals, sig_digits :
            Rundungseinstellungen für die angezeigten Werte.
        distance : float | None
            Abstand der Kraftpfeile vom Knoten. None = Standardabstand
            (wie bei normalen Lasten).
        show_text : bool, default=True
            Ob die Zahlenwerte der Auflagerkräfte angezeigt werden.

        Returns
        -------
        tuple[SystemGeo, list[PointLoadGeo]]
            Die Systemgeometrie und eine Liste der Auflagerkraft-Objekte.
        """
        forces = solver.system_support_forces                           # Vektor, 3 Werte je Knoten

        elements = []
        for index, node in enumerate(solver.nodes):
            if not cls._is_supported(node):
                continue

            px = float(forces[index * 3 + 0, 0])
            pz = float(forces[index * 3 + 1, 0])
            pm = float(forces[index * 3 + 2, 0])

            if px == 0.0 and pz == 0.0 and pm == 0.0:
                continue                                            # keine Reaktion vorhanden -> nichts zeichnen

            reaction_load = NodePointLoad(x=px, z=pz, phi=pm)

            elements.append(PointLoadGeo(
                (node.x, node.z), load=reaction_load,
                distance=distance, show_text=show_text,
                decimals=decimals, sig_digits=sig_digits,
                element_type='reaction',
                line_style=DEFAULT_STATE_LINE,
                text_style=DEFAULT_STATE_LINE_TEXT,
                scaling=global_scale,
            ))

        return elements
    
    @classmethod
    def from_solver(
            cls,
            solver,
            bar_mesh_type: str = 'bars',
            decimals: int = 2,
            sig_digits: int | None = None,
            distance: float | None = None,
            show_text: bool = True,
    ):
        """Erzeugt die Systemgeometrie (SystemGeo) UND die Auflagerkraft-
        Elemente. Wird von fo.plot_reactions() genutzt, wenn Reaktionskräfte
        eigenständig (ohne bestehende SystemGeo) geplottet werden sollen."""
        sys_geo = SystemGeo(solver.system, mesh_type=bar_mesh_type)
        elements = cls.elements_from_solver(
            solver, sys_geo.global_scale, decimals, sig_digits,
            distance, show_text
        )
        return sys_geo, elements