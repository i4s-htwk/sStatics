"""
Example 02:
Advanced Poleplan

This example demonstrates the Poleplan for a more complex multi-bar system
with several hinges. Bars 1–3 form a bending-stiff sub-chain that behaves as
a single rigid kinematic chain (i.e., these bars do not rotate relative to
each other). However, the sub-chain as a whole is free to move within the
overall system, making the system mechanically unstable (mechanism).

The Poleplan shows the rigid-body displacements of all kinematic chains.
By default, the rotation angle of the first chain is set to 1. Here, the
rotation of a specific chain (chain index 1) is manually set to 0.2 before
plotting. This illustrates how adjusting the rotation of individual chains
affects the mechanism’s overall displacement pattern.
"""


from sstatics.core.preprocessing import (
    Node, Bar, Material, CrossSection, System
)
from sstatics.core.solution import Poleplan

# 1. Define cross-section and material
c_1 = CrossSection(mom_of_int=2769, area=76.84, height=20, width=10,
                   shear_cor=0.1)
m_1 = Material(young_mod=21000, poisson=0.1, shear_mod=8100,
               therm_exp_coeff=0.1)

# 2. Define nodes (some supported, some free)
node_1 = Node(0, 0, u='fixed', w='fixed')
node_2 = Node(0, -3)
node_3 = Node(2.5, -3)
node_4 = Node(6.5, -3)
node_5 = Node(8, -3)
node_6 = Node(8, 0, u='fixed', w='fixed')

# 3. Define bars with hinges
bar_1 = Bar(node_1, node_3, c_1, m_1, hinge_phi_j=True)
bar_2 = Bar(node_1, node_2, c_1, m_1, hinge_phi_i=True, hinge_phi_j=True)
bar_3 = Bar(node_2, node_3, c_1, m_1, hinge_phi_j=True)
bar_4 = Bar(node_3, node_4, c_1, m_1, hinge_phi_j=True)
bar_5 = Bar(node_4, node_5, c_1, m_1)
bar_6 = Bar(node_5, node_6, c_1, m_1)

bars = [bar_1, bar_2, bar_3, bar_4, bar_5, bar_6]

# 4. Define system
system = System(bars)

# 5. Create poleplan
poleplan = Poleplan(system)

# 6. Plot rigid-body displacement
# -> rotation of first chain set to 1 by default
# -> Adjust rotation angle of a specific chain manually
poleplan.set_angle(chain_idx=1, angle=0.2)
poleplan.plot()

print("=== Advanced Poleplan Example ===")
print("Poleplan generated for multi-bar frame system showing rigid-body "
      "displacements of chains with a manually adjusted chain rotation.")
