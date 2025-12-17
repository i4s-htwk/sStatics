"""
Example 01:
Pole plan for a simple beam system

This example illustrates the use of the Poleplan class to determine the
rigid-body displacement of a statically indeterminate system that becomes
mechanically unstable (mechanism) after removing some supports. The Poleplan
provides a graphical representation of the kinematic chains.
"""

from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System
)
from sstatics.core.solution import Poleplan
from sstatics.core.postprocessing.graphic_objects import (
    ObjectRenderer, SystemGeo)


# 1. Define cross-section and material
c_1 = CrossSection(mom_of_int=2769, area=76.84, height=20, width=10,
                   shear_cor=0.1)
m_1 = Material(young_mod=21000, poisson=0.1, shear_mod=8100,
               therm_exp_coeff=0.1)

# 2. Define nodes (some with supports, some free)
node_1 = Node(0, 0, u='fixed', w='fixed')
node_2 = Node(200, 0)
node_3 = Node(400, 0, w='fixed')
node_4 = Node(600, 0)

# 3. Define bars
bar_1 = Bar(node_1, node_2, c_1, m_1, hinge_phi_j=True)
bar_2 = Bar(node_2, node_3, c_1, m_1)
bar_3 = Bar(node_3, node_4, c_1, m_1)

bars = [bar_1, bar_2, bar_3]

# 4. Define system
system = System(bars)

# Visualize system
ObjectRenderer(SystemGeo(system, show_bar_text=True), 'plotly').show()

# 5. Create poleplan and plot rigid-body displacements
poleplan = Poleplan(system)
poleplan.plot()

print("=== Pole Plan Example ===")
print("Poleplan generated for the beam system, showing rigid-body displacement"
      "of chains.")
