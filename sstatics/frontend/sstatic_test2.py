from sstatics.core.preprocessing.geometry import Polygon

from sstatics.core.preprocessing import (Material, Node, Bar, System,
                                         CrossSection, BarPointLoad)

from sstatics.core.solution import FirstOrder
from sstatics.core.postprocessing import SystemResult

# from sstatics.graphic_objects import ResultGraphic, CrossSectionGraphic
from sstatics.graphic_objects.stress import CrossSectionStressGraphic

# import numpy as np

# -------------------------- Cross-Section --------------------------------- #
cross_sec = CrossSection(
    # 0.0001126, 0.0106, 0.2, 0.1, 0.1
    geometry=[Polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)])]
)

# Visualize the cross-section
# CrossSectionGraphic(cross_section=cross_sec, merged=True).show()


# -------------------------- Material  ------------------------------------- #
# Define material: E-Modulus = 11,000,000 kN/m²
material = Material(11000000, 0.1, 0.1, 0.1)


# ----------------------------- Model -------------------------------------- #
# Define nodes
n1 = Node(0, 0, u='fixed', w='fixed')
n2 = Node(3, 0, w='fixed')
n3 = Node(4, 0)

# Define bar connecting Nodes
bar_model_1 = Bar(n1, n2, cross_sec, material,
                  point_loads=[BarPointLoad(x=10, z=10, position=0.8)])

bar_model_2 = Bar(n2, n3, cross_sec, material)

# Create system
system = System([bar_model_1, bar_model_2])

# divide Bar
system.create_mesh({bar_model_1: [0.5]})

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
    n_disc=1,)

bar_result = results.bars
deforms, forces = solution.calc

# ----------------------------- Schnittgrößen ------------------------------ #
for i, f in enumerate(forces):         # gibt für jeden Teilstab
    # einzeln die Schnittkräfte aus
    print("----------------Teilstab ", i, '---------------')
    print(f)

# ----------------------------- Spannungen --------------------------------- #
for i, b in enumerate(bar_result):
    n = b._normal_stress
    v_max = b._shear_stress_max
    v_disc = b._shear_stress_disc
    m = b._bending_stress
    print("----------------Teilstab ", i, '---------------')
    print('Normalspannung:', n)
    print('Maximale Schubspannung: ', v_max)
    print('Schubspannungen entlang des Querschnittes: ', v_disc)
    print('Biegespannung:', m)

# ----------------------------- Plot --------------------------------------- #
# # Plot vertical deflection
# ResultGraphic(results,
#               'shear',
#               bar_mesh_type='bars',
#               result_mesh_type='bars'
#               ).show()

CrossSectionStressGraphic(
    system_result=results,
    bar=bar_model_1,
    position=0.8,
    side='left',
    kind=['shear', 'bending', 'normal'],
).show()
