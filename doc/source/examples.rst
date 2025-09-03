=====================
Example 1: Node Deflection
=====================

We want to model a simple cantilever beam, apply a vertical point load
at its free end, and calculate the vertical deflection.
This example is designed as a beginner-friendly step-by-step guide,
with clear instructions and expected outputs.

**System description**

- Beam length: 3 m
- Material: Timber C24, :math:`E \> = \> 11.000.000 \dfrac{kN}{m^2}`
- Cross-section: width / height = :math:`10\> cm / 20 \> cm`
- Support: fixed at the left end (Node 1)
- Load: :math:`1\> kN` vertical point load at the free end (Node 2)

.. image:: images/example1_system.png
   :alt: Cantilever beam system

.. note::
   All values are converted into **meters (m)** and **kilonewtons (kN)**
   to keep the units consistent.


Step 1: Define Cross-Section and Material
-----------------------------------------

Instead of providing stiffness values directly, we create a **polygon shape**
for the cross-section. The program automatically calculates its section properties.

The polygon points are:
P1 = (0, 0), P2 = (0.10, 0), P3 = (0.10, 0.20), P4 = (0, 0.20)

.. code-block:: python

    from core.preprocessing import CrossSection, Polygon
    from graphic_objects import CrossSectionGraphic

    # Define rectangular cross-section using coordinates
    cross_sec = CrossSection(
        geometry=[Polygon([(0, 0), (0.1, 0), (0.1, 0.2), (0, 0.2), (0, 0)])]
    )

    # Visualize the cross-section
    CrossSectionGraphic(cross_section=cross_sec).show()

.. image::
    images/example1_cross_section.png

Now we define the material properties:

.. code-block:: python

    from core.preprocessing import Material

    # Define material: E-Modulus = 11,000,000 kN/m²
    material = Material(11000000, 0.1, 0.1, 0.1)


Step 2: Define Nodes
---------------------

The beam is defined by two nodes:

- **Node 1**: Fixed support (at position 0, 0)
- **Node 2**: Free end (at position 3, 0) with a vertical load of 1 kN

.. code-block:: python

    from core.preprocessing import Node, NodePointLoad

    # Define nodes
    n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')   # Fixed support
    n2 = Node(3, 0, loads=NodePointLoad(z=1))            # Loaded free end


Step 3: Define Bar
-------------------

Now we connect the two nodes with a bar element.

.. code-block:: python

    from core.preprocessing import Bar

    # Define bar connecting Node 1 and Node 2
    bar = Bar(n1, n2, cross_sec, material)


Step 4: Assemble and Solve the System
--------------------------------------

We assemble the bar into a structural system and perform a
first-order analysis.

.. code-block:: python

    from core.preprocessing import System
    from core.solution import FirstOrder

    # Create system
    system = System([bar])

    # Perform calculation
    solution = FirstOrder(system)

    # Get bar deformations
    print(solution.bar_deform_list)

**Expected output (simplified):**

.. code-block::

    [array([[ 0.          ],
            [ 0.          ],
            [ 0.          ],
            [ 0.          ],
            [0.0122727273 ],
            [-0.0061363636]])]

**Explanation of results:**

* The array contains **6 values**:
    * First 3 values = displacements :math:`(u, w, \varphi)` at the *start node* (Node 1)
    * Last 3 values = displacements :math:`(u, w, \varphi)` at the *end node* (Node 2)

* Interpretation for our case:
    * Node 1 is fixed → all values = 0
    * Node 2 has:
        * Horizontal displacement :math:`u = 0`
        * **Vertical displacement w = 12.27 mm**
        * Rotation :math:`\varphi = -0.0061\> rad`


Step 5: Visualize Results
--------------------------

Finally, we plot the deflected shape of the beam.
This gives a graphical overview of the displacement under the applied load.

.. code-block:: python

    from postprocessing import SystemResult
    from graphic_objects import ResultGraphic

    # Prepare results for plotting
    results = SystemResult(
        system,
        solution.bar_deform_list,
        solution.internal_forces,
        solution.node_deform,
        solution.node_support_forces,
        solution.system_support_forces
    )

    # Plot vertical deflection
    ResultGraphic(results, 'w').show()

.. image::
    images/example1_result.png


**Result:**
The diagram shows the beam deflecting downward at Node 2.
The calculated value of **12.27 mm vertical deflection** is the key result of this example.
