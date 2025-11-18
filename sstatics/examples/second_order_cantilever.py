# Second Order: Cantilever Colum

# Import Modules
from sstatics.core.preprocessing import (Bar, BarLineLoad, CrossSection,
                                         Material, Node, NodePointLoad, System)
from sstatics.core.calc_methods import FirstOrder, SecondOrder

# Define Cross-Section and Material
c_1 = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
m_1 = Material(210000000, 0.1, 81000000, 0.1)

# Define Nodes
node_1 = Node(x=0, z=0, u='fixed', w='fixed', phi='fixed')
node_2 = Node(x=0, z=-4, loads=NodePointLoad(x=0, z=182, phi=0, rotation=0))

# Define Bar
bar_1 = Bar(node_1, node_2, c_1, m_1, line_loads=BarLineLoad(
    pi=1, pj=1.5, direction='z', coord='bar', length='exact'))

# Define System
system = System([bar_1])

# First Order Analysis
solution_first_order = FirstOrder(system)

bar_deform_first = solution_first_order.bar_deform_list
forces_first = solution_first_order.internal_forces

# Visulize Results
print(bar_deform_first)
print(forces_first)

# Solver Approaches

# Create a SecondOrder Object
sec_order = SecondOrder(system)

# Run matrix approach
sec_order.matrix_approach('analytic')
solution_analytic = sec_order.solver_matrix_approach

# Comparing the bending moment at the beginning of the bar
sec_order.matrix_approach('taylor')
solution_taylor = sec_order.solver_matrix_approach
sec_order.matrix_approach('p_delta')
solution_p_delta = sec_order.solver_matrix_approach

# Comparing the internal forces of the bending moment at the beginning of the
# bar
print('Mi - first order: [kNm]', forces_first[0][2][0])
print('Mi - second order analytic: [kNm]',
      solution_analytic.internal_forces[0][2][0])
print('Mi - second order taylor: [kNm]',
      solution_taylor.internal_forces[0][2][0])
print('Mi - second order p delta: [kNm]',
      solution_p_delta.internal_forces[0][2][0])

# Comparing the bar deformation at the end of the cantilever in w - direction
# Comparing the bar deformation at the end of the cantilever
print('w - first order: [m]', bar_deform_first[0][4][0])
print('w - second order analytic: [m]',
      solution_analytic.bar_deform_list[0][4][0])
print('w - second order taylor: [m]',
      solution_taylor.bar_deform_list[0][4][0])
print('w - second order p delta: [m]',
      solution_p_delta.bar_deform_list[0][4][0])
