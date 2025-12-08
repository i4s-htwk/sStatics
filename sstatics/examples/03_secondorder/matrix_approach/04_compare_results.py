"""
Example 04:
Comparison of the Bending Moment at the Start of the Bar

This example provides a comparison of the bending moment at the fixed end of
the bar. For this purpose, the bending moments obtained from the second-order
calculations using the various matrix approaches are compared with the bending
moment determined according to first-order theory.
"""
from sstatics.core.preprocessing import (Bar, BarLineLoad, CrossSection,
                                         Material, Node, NodePointLoad, System)
from sstatics.core.calc_methods import SecondOrder, FirstOrder

# 1. Define cross-section and material -> steel and HEA 240 profile
c_1 = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
m_1 = Material(210000000, 0.1, 81000000, 0.1)

# 2. Define nodes (cantilever system)
node_1 = Node(x=0, z=0, u='fixed', w='fixed', phi='fixed')
node_2 = Node(x=0, z=-4, loads=NodePointLoad(x=0, z=182, phi=0, rotation=0))

# 3. Define bar
bar_1 = Bar(node_1, node_2, c_1, m_1, line_loads=BarLineLoad(
    pi=1, pj=1.5, direction='z', coord='bar', length='exact'))

# 4. Define system
system = System([bar_1])

# 5. Create a SecondOrder object
sec_order = SecondOrder(system)

# 6. Set matrix approach and a chosen variant and get solver object
# analytic
sec_order.matrix_approach('analytic')
solution_analytic = sec_order.solver_matrix_approach

# taylor
sec_order.matrix_approach('taylor')
solution_taylor = sec_order.solver_matrix_approach

# p-delta
sec_order.matrix_approach('p_delta')
solution_p_delta = sec_order.solver_matrix_approach

# first order
solution_first_order = FirstOrder(system)

# 7. Get moment at the beginning of the bar
moment_analytic = solution_analytic.internal_forces[0][2][0]
moment_taylor = solution_taylor.internal_forces[0][2][0]
moment_p_delta = solution_p_delta.internal_forces[0][2][0]
moment_first = solution_first_order.internal_forces[0][2][0]

print("=== Second Order Example - comparing results===")
print(f'Mi - first order: {moment_first} [kNm]')
print(f'Mi - second order analytic: {moment_analytic} [kNm]')
print(f'Mi - second order taylor: {moment_taylor} [kNm]')
print(f'Mi - second order p delta: {moment_p_delta} [kNm]')
