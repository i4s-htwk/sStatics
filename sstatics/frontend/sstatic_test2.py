from sstatics.core.preprocessing.geometry import Polygon

from sstatics.core.preprocessing import (Material, Node, Bar, System,
                                         CrossSection, BarPointLoad,
                                         BarLineLoad)

from sstatics.core.solution import FirstOrder
from sstatics.core.postprocessing import SystemResult

from sstatics.graphic_objects.stress import CrossSectionStressGraphic
from sstatics.graphic_objects import SystemResultGraphic
# import numpy as np

# -------------------------- Cross-Section --------------------------------- #
cross_sec = CrossSection(
    # 0.003125, 0.15, 0.5, 0.3, 0.1
    geometry=[
        # Polygon([(-0.1, -0.2), (0.1, -0.2),
        # (0.2, 0.2), (-0.2, 0.2), (-0.1, -0.2)])]
        Polygon([(-1, -2), (1, -2), (2, 2), (-2, 2), (-1, -2)])]
    # [Polygon([(6, 7), (6, 5.5), (-6, 5.5), (-6, 7), (6, 7)]),
    # Polygon([(0.5, 5.5), (0.5, -5.5), (-0.5, -5.5), (-0.5, 5.5),
    # (0.5, 5.5)]),
    # Polygon([(6, -7), (6, -5.5), (-6, -5.5), (-6, -7), (6, -7)])]
)

# -------------------------- Material  ------------------------------------- #
# Define material: E-Modulus = 11,000,000 kN/m²
material = Material(11000000, 0.1, 0.1, 0.1)

# ----------------------------- Model -------------------------------------- #
# Define nodes
n1 = Node(0, 0, u='fixed', w='fixed')
n2 = Node(3, 0, w='fixed')
n3 = Node(4, 0)

# Define loads
point_load_1 = BarPointLoad(x=10, z=30, position=0.8)
line_load_1 = BarLineLoad(pi=10, pj=10, direction='z', coord='bar',
                          length='exact')
line_load_2 = BarLineLoad(pi=5, pj=5, direction='z', coord='bar',
                          length='exact')

# Define bar connecting Nodes
bar_1 = Bar(n1, n2, cross_sec, material,
            point_loads=[point_load_1],
            # line_loads=[line_load_1]
            )

bar_2 = Bar(n2, n3, cross_sec, material,
            line_loads=[line_load_2]
            )

# Create system
system = System([bar_1, bar_2])

# divide Bar
system.create_mesh({bar_1: [0.5]})

# -------------------------- Analysis  ------------------------------------- #
# Perform calculation
solution = FirstOrder(system)

results = SystemResult(
    system,
    solution.bar_deform_list,
    solution.internal_forces,
    solution.node_deform,
    solution.node_support_forces,
    solution.system_support_forces,
    n_disc=10,)

bar_result = results.bars
deforms, forces = solution.calc

# ----------------------------- Schnittgrößen ------------------------------ #
# for i, f in enumerate(forces):
#     print('----------------Teilstab ', i, '---------------')
#     print(f)

# ----------------------------- Spannungen --------------------------------- #
for i, b in enumerate(bar_result, start=1):
    print("----------------Teilstab ", i, '---------------')
    n = b._normal_stress
    v_max = b._shear_stress_max
    m = b._bending_stress
    n_disc = b.normal_stress_disc
    v_disc = b.shear_stress_disc
    m_disc = b.bending_stress_disc
    # print('Normalspannung (Anfang/Ende)', n)
    # print('Normalspannung:', n_disc)
    # print('Maximale Schubspannung (Anfang/Ende): ', v_max)
    # print('Maximale Schubspannung: ', v_disc)
    # print('Biegespannung (Anfang/Ende)', m)
    # print('Biegespannung:', m_disc)


# ----------------------------- Plot --------------------------------------- #
SystemResultGraphic(
    system_result=results,
    kind='shear_stress',
    mesh_type='mesh'
).show()

CrossSectionStressGraphic(
    system_result=results,
    bar=bar_1,
    position=0.8,
    side='left',
    kind=['normal', 'bending', 'shear'],
    discretization=20
).show()

CrossSectionStressGraphic(
    system_result=results,
    bar=bar_1,
    position=0.8,
    side='right',
    kind=['normal', 'bending', 'shear'],
    discretization=20
).show()
