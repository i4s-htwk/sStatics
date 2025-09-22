# from ctypes import kind

import numpy as np

from functools import cached_property
from typing import Literal, Tuple, List, Any, Union

from sstatics.core.preprocessing import Bar, CrossSection
from sstatics.core.postprocessing.results import SystemResult

from sstatics.graphic_objects.utils import SingleGraphicObject, transform
from sstatics.graphic_objects.geometry import LineGraphic
from sstatics.graphic_objects.cross_section import CrossSectionGraphic


class CrossSectionStressGraphic(SingleGraphicObject):

    def __init__(
            self,
            system_result: SystemResult,
            bar: Bar,
            position: float,
            side: Literal['left', 'right'],
            kind: Union[
                Literal['normal', 'shear', 'bending'],
                List[Literal['normal', 'shear', 'bending']]
            ] = 'normal',
            **kwargs
    ):
        if not isinstance(system_result, SystemResult):
            raise TypeError(
                '"system_result" has to be an instance of SystemResult'
            )
        super().__init__(
            system_result.bars[0].bar.node_i.x,
            system_result.bars[0].bar.node_i.z,
            **kwargs
        )

        if isinstance(kind, str):   # falls kind ein einzelner Wert
            kind = [kind]           # kind in Liste umwandeln
        self.kind = kind

        self.system = system_result.system
        mesh_segments = self.system.mesh_segments_of(bar)

        # calculates point of interest
        c, s = np.cos(bar.inclination), np.sin(bar.inclination)
        point = (
            bar.node_i.x + c * position * bar.length,
            bar.node_i.z - s * position * bar.length
        )
        print('Gewünschter Point: ', point)

        # checks which bar the point belongs to and at which side (i or j)
        #   distincts if the bar is defined from left to right or right to left
        if bar.node_i.x < bar.node_j.x:
            # bar is defined from left to right
            for i, seg in enumerate(mesh_segments):
                if (seg.node_i.x, seg.node_i.z) == point:
                    print('Gefundener Point:', (seg.node_i.x, seg.node_i.z))
                    print('ja, Anfangspunkt (i) von Stab ', i,
                          ' entspricht gesuchtem Punkt')
                    self.bar_of_interest = seg
                    self.barend_of_interest = 'i'
                    break
                elif (seg.node_j.x, seg.node_j.z) == point:
                    print('Gefundener Point:', (seg.node_j.x, seg.node_j.z))
                    print('ja, Endpunkt (j) von Stab ', i,
                          ' entspricht gesuchtem Punkt')
                    self.bar_of_interest = seg
                    self.barend_of_interest = 'j'
                    break
        else:
            # bar is defined from right to left
            for i, seg in enumerate(mesh_segments):
                print(i)
                if (seg.node_i.x, seg.node_i.z) == point:
                    print('Gefundener Point:', (seg.node_i.x, seg.node_i.z))
                    print('ja, Anfangspunkt (i) von Stab ', i,
                          ' entspricht gesuchtem Punkt')
                    self.bar_of_interest = seg
                    self.barend_of_interest = 'i'
                    break
                elif (seg.node_j.x, seg.node_j.z) == point:
                    print('Gefundener Point:', (seg.node_j.x, seg.node_j.z))
                    print('ja, Endpunkt (j) von Stab ', i,
                          ' entspricht gesuchtem Punkt')
                    self.bar_of_interest = seg
                    self.barend_of_interest = 'j'
                    break
        if self.bar_of_interest is None:
            # checks if the break was part of one of the segments
            raise ValueError(
                'An der gewünschten Position liegen keine Ergebnisse vor. '
                'Gib eine andere Position an, oder generiere ein passendes '
                'mesh.')

        # cross-section at the point of interest
        cross_section = self.bar_of_interest.cross_section

        # index of the bar of interest
        index = self.system.mesh.index(self.bar_of_interest)
        print(index)

        # system_results at the point of interest
        bar_result = system_result.bars[index]

        # Check, if i or j is needed, to pick the right column
        column = 0 if self.barend_of_interest == 'i' else 1

        # Create a list to fill with the stresses wanted by user
        which_stress = []

        # Fills list 'which_stress' depending on the given kind(s)
        if 'normal' in kind:
            n_stress = bar_result._normal_stress[:, column]
            print('Normalspannung: ', n_stress)
            which_stress.append((n_stress, cross_section, 'normal', 'blue'))
        if 'bending' in kind:
            m_stress = bar_result._bending_stress[:, column]
            print('Biegemomentspannung: ', m_stress)
            which_stress.append((m_stress, cross_section, 'bending', 'green'))
        if 'shear' in kind:
            v_stress = bar_result._shear_stress_disc[:, column]
            print('Schubspannung: ', v_stress)
            which_stress.append((v_stress, cross_section, 'shear', 'red'))

        # Check if 'which_stress' is not empty
        if not which_stress:
            raise ValueError(
                '"kind" must contain at least one of: "normal", "shear", '
                '"bending"'
            )

        # Calculate total maximum stress-value
        max_vals = [np.max(np.abs(vals)) for (vals, _, _, _) in which_stress]
        global_max = np.max(max_vals)

        # Create a list to fill with the StressGraphic-objects
        self._stress_plot = []

        # Fill the list '_stress_plot' with StressGraphic-objects
        for stress, cross_section, kind, color in which_stress:
            self._stress_plot.append(
                StressGraphic(
                    stress=stress,
                    cross_section=cross_section,
                    kind=kind,
                    color=color,
                    max_value=global_max
                )
            )

        self._cross_section = [
            CrossSectionGraphic(cross_section=cross_section),
        ]

    @property
    def annotations(self):
        return tuple(
            ann for rg in self._stress_plot for ann in rg.annotations
        )

    @property
    def traces(self):
        traces = []

        for cs in self._cross_section:
            traces.extend(
                cs.transform_traces(self.x, self.z, self.rotation, self.scale)
            )
        for i, st in enumerate(self._stress_plot):
            traces.extend(
                st.transform_traces(
                    self.x, self.z,
                    self.rotation,
                    self.scale,
                    translation=(i * self.bar_of_interest.cross_section.width,
                                 0)
                )
            )
        return traces


class StressGraphic(SingleGraphicObject):

    def __init__(self, stress, cross_section: CrossSection, kind,
                 color,
                 decimals: (int | None) = None,
                 sig_digits: int | None = None,
                 base_scale=None,
                 max_value=None,
                 **kwargs
                 ):
        super().__init__(
            # Einfügepunkt
            cross_section.center_of_mass_y, cross_section.center_of_mass_z,
            **kwargs
        )

        self.stress = stress
        self.cross_section = cross_section
        self.height = self.cross_section.height
        self.width = self.cross_section.width
        self.kind = kind
        self.color = color
        self.decimals = decimals
        self.sig_digits = sig_digits
        self.base_scale = base_scale
        self.max_value = max_value

    @cached_property
    def _bound_results(self):
        if self.kind in ('normal', 'bending'):
            print('TEST', self.stress[0], self.stress[-1])
            return self.stress[0], self.stress[-1]
        else:
            print('TEST', self.stress[0], np.max(self.stress), self.stress[-1])
            return self.stress[0], np.max(self.stress), self.stress[-1]

    def _round_value(self, value):
        # rounds the shown value to two decimals
        if self.decimals is not None:
            return np.round(value, self.decimals)
        if self.sig_digits is not None:
            return f"{value:.{self.sig_digits}g}"
        return np.round(value, 2)

    # TODO
    @cached_property
    def _annotation_pos(self):
        # defines the position of the annotation
        # TODO d definieren
        # TODO bei shear: max etwas nach rechts verschieben
        d = 0.15 * 1
        if self.kind in ('normal', 'bending'):
            return [
                (self.stress[0], (0 - d)),
                (self.stress[1], (self.height + d))
            ]
        else:
            return [
                (self.stress[0], (0 - d)),
                ((np.max(self.stress)), self.height/2),
                (self.stress[-1], (self.height + d))
            ]

    @cached_property
    def _annotation_text(self) -> Tuple[Any | str, ...]:
        return tuple(
            self._round_value(r) for r in self._bound_results
        )

    @cached_property
    def _max_value(self):
        # Return the total maximum stress-value
        if self.max_value:
            return self.max_value
        else:
            max_val = np.max(np.abs(self.stress))
            return 1e-6 if np.isclose(max_val, 0) else max_val

    @cached_property
    def _base_scale(self):
        # if no base_scale given (=None), the scale is based on the
        # width of the cross-section
        return self.base_scale / self._max_value if self.base_scale \
            else (self.width/2) / self._max_value

    # TODO Translation für kombinierte kind anpassen
    @property
    def _annotations(self):
        y, z = np.array(list(zip(*self._annotation_pos)))
        y, z = transform(
            ox=0, oz=0,
            x=(y * self._base_scale), z=z,
            scale=self.scale,
            translation=(1.5 * self.width, 0)
        )
        if self.kind in ('normal', 'bending'):
            return (
                (y[0], z[0], self._annotation_text[0]),
                (y[1], z[1], self._annotation_text[1])
            )
        else:
            return (
                (y[0], z[0], self._annotation_text[0]),
                (y[1], z[1], self._annotation_text[1]),
                (y[2], z[2], self._annotation_text[2])
            )

    @property
    def create_points(self):
        # Return the points to create the stress distribution
        # Includes 'base_scale' to scale the distribution to the size of the cs
        if self.kind in ('normal', 'bending'):
            # normal and bending stress are linear and can be plotted
            # by only 2 stress-values
            points = [(0, 0),
                      (self.stress[0] * self._base_scale, 0),
                      (self.stress[1] * self._base_scale, self.height),
                      (0, self.height),
                      (0, 0)]
        else:
            # shear stress is parabolic and is plotted by n_disc+1 points to be
            # more accurate
            z = self.stress * self._base_scale
            y = self.cross_section.height_disc
            points = list(zip(z, y))
            points.append((0, 0))

        return points

    @property
    def traces(self):
        # Create the stress distribution as polygon from given points
        points = self.create_points
        return (LineGraphic.from_points(
            points,
            scatter_options=self.scatter_kwargs | {'line_color': self.color}
        ).transform_traces(0, 0, 0, translation=(1.5 * self.width, 0))
        )
