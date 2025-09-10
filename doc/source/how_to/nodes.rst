Nodes
=====
This section covers practical tasks for working with nodes in the structural system.
It includes creating nodes, applying loads and displacements, and handling rotations
and elastic supports.

Create a Fixed Node
-------------------

In many models, you need to define a support that prevents all
translations and rotations of a node. For example, a fixed support
at the beginning of a cantilever beam.

This can be done by setting the degrees of freedom ``u``, ``w`` and
``phi`` to ``"fixed"``.

.. code-block:: python

    from sstatics.core.preprocessing import Node

    # Create a node at the origin with all DOFs fixed
    n1 = Node(x=0, z=0, u="fixed", w="fixed", phi="fixed")


If you plot the system, the fixed node will be displayed as:

.. code-block:: python

    from sstatics.graphic_objects import NodeGraphic

    # Plot the created node
    NodeGraphic(n1).show()

.. image:: images/node_fixed.png
   :alt: A node with spring supports in x and z direction
   :width: 200px
   :align: center

Create a Free Node
------------------

A free node has no constraints. This means all three degrees of freedom
(``u``, ``w``, ``phi``) are set to ``"free"``.

This type of node is typically used at the end of a cantilever beam or
as an intermediate node without support.

.. code-block:: python

    from sstatics.core.preprocessing import Node

    # Create a free node at position (5, 0)
    n2 = Node(x=5, z=0)

By default, all degrees of freedom are set to ``"free"``,
so you do not need to specify them explicitly.

Create a Spring (Elastic) Support Node
--------------------------------------

Instead of being fully fixed or completely free, a node can also be
supported by an elastic spring. This is useful to model foundations
or connections with finite stiffness.

You define the stiffness value directly
as a number for ``u``, ``w`` or ``phi``.

.. code-block:: python

    from sstatics.core.preprocessing import Node

    # Create a node at the origin with:
    # - horizontal spring support (stiffness = 100)
    # - vertical spring support (stiffness = 2000)
    # - free rotation
    n3 = Node(x=0, z=0, u=100, w=2000, phi="free")

    from sstatics.graphic_objects import NodeGraphic
    # Plot the created node
    NodeGraphic(n3).show()

.. image:: images/spring_node.png
   :alt: A node with spring supports in x and z direction
   :width: 400px
   :align: center

**Plot not possible in new sstatics!!!**

.. note::

   The program does not enforce a specific unit system.
   This means that you must ensure **consistency** between your inputs.
   For example, if you define stiffness values in :math:`\text{kN}/\text{m}`,
   then all related quantities (forces, displacements, cross-sections) must
   be provided in the same unit system.


Rotate a Node
-------------

Nodes can be rotated.

.. code-block:: python

    import numpy
    from sstatics.core.preprocessing import Node
    # Node not rotated
    n1 = Node(x=0, z=0, u='fixed', w='fixed')
    # Node rotated
    n2 = Node(x=0, z=0, u='fixed', w='fixed', rotation=numpy.pi/4)

If you plot the nodes, they will be displayed as:

.. code-block:: python

    from sstatics.graphic_objects import NodeGraphic

    # Plot nodes
    NodeGraphic(n1).show()
    NodeGraphic(n2).show()

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: images/not_rotated_node.png
         :alt: Node without rotation
         :width: 200px

     - .. figure:: images/rotated_node.png
         :alt: Rotated Node
         :width: 200px

Apply a Single Point Load to a Node
-----------------------------------

A point load acts directly on a node. Define its components in the x, z,
and phi directions using the :class:`NodePointLoad` class.

.. code-block:: python

    from sstatics.core.preprocessing import Node, NodePointLoad

    load = NodePointLoad(x=0, z=10, phi=0)
    n1 = Node(x=0, z=0, loads=load)

Apply Multiple Point Loads to a Node
------------------------------------

Multiple point loads can be applied simultaneously by passing a list or tuple
to the ``loads`` parameter. The program sums all loads internally.

.. code-block:: python

    import numpy
    from sstatics.core.preprocessing import Node, NodePointLoad

    load1 = NodePointLoad(x=5, z=0, phi=0)
    # Define a NodeLoad, which is rotated by 90 degrees
    load2 = NodePointLoad(x=0, z=-10, phi=0, rotation=numpy.pi/2)
    n1 = Node(x=0, z=0, loads=[load1, load2])

Prescribe a Displacement at a Node
----------------------------------

Prescribed displacements define nodal settlements or rotations.
Use the :class:`NodeDisplacement` class to set a displacement in x, z,
or rotational direction phi.

.. code-block:: python

    from sstatics.core.preprocessing import Node, NodeDisplacement

    disp = NodeDisplacement(x=0.01, z=0, phi=0)
    n1 = Node(x=0, z=0, displacements=disp)

Apply Multiple Displacements at a Node
--------------------------------------

Multiple prescribed displacements can be combined in a list or tuple.
The program automatically sums all displacements.

.. code-block:: python

    from sstatics.core.preprocessing import Node, NodeDisplacement

    disp1 = NodeDisplacement(x=0.01, z=0, phi=0)
    disp2 = NodeDisplacement(x=-0.02, z=0.005, phi=0)
    n1 = Node(x=0, z=0, displacements=[disp1, disp2])
