# Poleplan

# Import Modules
from sstatics.core.preprocessing import (Node, Bar, Material, CrossSection,
                                         System)
from sstatics.graphic_objects import SystemGraphic
from sstatics.graphic_objects.poleplan import PoleplanGraphic

# define cross-section
c_1 = CrossSection(mom_of_int=2769, area=76.84, height=20, width=10,
                   shear_cor=0.1)

# define material
m_1 = Material(young_mod=21000, poisson=0.1, shear_mod=8100,
               therm_exp_coeff=0.1)

# define Node
node_1 = Node(0, 0, u='fixed', w='fixed')
node_2 = Node(200, 0)
node_3 = Node(400, 0, w='fixed')
node_4 = Node(600, 0)

# define Bar
bar_1 = Bar(node_1, node_2, c_1, m_1, hinge_phi_j=True)
bar_2 = Bar(node_2, node_3, c_1, m_1)
bar_3 = Bar(node_3, node_4, c_1, m_1)

bars = [bar_1, bar_2, bar_3]

# define System
system = System(bars)

# visualize system
SystemGraphic(system=system).show()

# Define Poleplan
poleplan = system.polplan
poleplan.set_angle(target_chain=poleplan.chains[0], target_angle=0.2)

# Visualize Poleplan
PoleplanGraphic(poleplan=poleplan, show_annotations=False).show()
