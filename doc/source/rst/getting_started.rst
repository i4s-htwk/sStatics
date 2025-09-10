===============
Getting Started
===============

sStatics is divided into three main stages:

**Pre-processing**: modeling the structural system and defining system parameters

**Solution**: running the chosen analysis method

**Post-processing**: visualizing results, computing coefficients of differential equations,
and extracting intermediate values (e.g., from virtual work, reduction theorem, or the force method)
to support verification of hand calculations.

Example
=======

To demonstrate the workflow, let us build a small example project.

The example system is a **cantilever beam** subjected to a uniform
line load. The goal is to calculate the internal forces of the beam and
then display the corresponding diagrams.

.. image::
    images/tutorial1.png

Pre-Processing
==============

Pre-processing defines the foundation of the structural model.

**Step 1: Define material and cross-section**

In our example, we use a HEA 240 profile with steel S235.

.. code-block:: python

    from sstatics.core.preprocessing import CrossSection, Material

    material = Material(210000000, 0.1, 81000000, 0.1)
    cross_sec = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.1)

**Step 2: Define nodes**

A bar is defined by two nodes. Each node requires coordinates and support
conditions.

- The first node is fixed (all degrees of freedom ``u``, ``w``, ``phi`` set to
  "fixed").
- The second node is free (default = "free").

.. code-block:: python

    from sstatics.core.preprocessing import Node

    n1 = Node(x=0, z=0, u='fixed', w='fixed', phi='fixed')
    n2 = Node(x=5, z=0)

**Step 3: Define loads**

The cantilever is subjected to a uniform line load in the *z*-direction.
This is represented by a :py:attr:`BarLineLoad` object.

.. code-block:: python

    from sstatics.core.preprocessing import BarLineLoad

    load = BarLineLoad(pi=1, pj=1, direction='z', coord='bar', length='exact')

**Step 4: Create bar and system**

Finally, we assemble the bar with the defined attributes and build the
structural system as a collection of bars.

.. code-block:: python

    from sstatics.core.preprocessing import Bar, System

    bar = Bar(n1, n2, cross_sec, material, line_loads=load)
    system = System([bar])

Solution
========

To compute the internal forces of the system, we pass the model to the
``FirstOrder`` solver. The method ``internal_forces`` returns the
calculated forces.

.. code-block:: python

    from sstatics.core.solution import FirstOrder

    fo = FirstOrder(system)
    forces = fo.internal_forces

Post-Processing
===============

With the solution complete, the results can be visualized by using ``ResultGraphic``.
In our example, we want to display the **bending moment diagram** of the system.
By setting the parameter ``kind="moment"``, the bending moment distribution
along the beam is plotted.

.. code-block:: python

        from sstatics.core.postprocessing import SystemResult
        from sstatics.graphic_objects import ResultGraphic

        results = SystemResult(system, fo.bar_deform_list, fo.internal_forces, fo.node_deform, fo.node_support_forces, fo.system_support_forces)
        ResultGraphic(results, 'moment').show()

.. image:: images/tutorial_moment.png
   :alt: Resulting moment forces
   :align: center

You have now completed your first analysis: a cantilever beam subjected
to a uniform line load. This workflow, **Pre-processing → Solution →
Post-processing**, is the basic structure for all further projects.
