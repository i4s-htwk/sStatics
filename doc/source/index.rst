
.. toctree::
    :hidden:
    :maxdepth: 2

    installation
    getting_started
    howtoguide
    examples
    explanation
    api

.. toctree::
    :caption: Development
    :hidden:

    contributing
    License <license>

========
sStatics
========
`sStatics` is a Python package for the analysis of planar truss structures
using the displacement method. `sStatics` can be used to calculate
displacements,  internal forces, and support reactions, and to visualise
structural behavior under applied loads. The package supports second-order
analysis and influence lines for advanced structural assessment.

Installation
------------
Download and install the current version of `sStatics`:

.. code:: console

   python -m pip install git+https://github.com/i4s-htwk/sStatics.git

See :ref:`label-installation` for more information.

License
-------
This project is licensed under the :doc:`GNU Affero General Public License
v3.0 (AGPL-3.0) <license>` .


Support
-------
Found a bug 🐛, or have a feature request ✨, raise an issue on the
`issue tracker <https://github.com/i4s-htwk/sStatics/issues>`_.
Alternatively you can get support on the
`discussions page <https://github.com/orgs/i4s-htwk/discussions>`_.

Disclaimer
----------
`sStatics` is an open-source engineering tool originally developed within a
research project at HTWK Leipzig and is now maintained and further
developed by the i4s institute. While care has been taken to implement the
underlying structural mechanics and numerical methods correctly—including
the displacement method, second-order effects, and influence line
calculations — it remains the user's responsibility to verify and assess all
results independently. The authors and contributors do not assume liability
for any incorrect use or interpretation of the software. Refer to the
license for details regarding permitted use and limitations.

Acknowledgements
----------------
This project is developed at the HTWK Leipzig University of Applied
Sciences and funded by the "Stiftung Innovation in der Hochschullehre".

.. image:: images/FAssMII_logo_WHITE_transparent.png
    :class: only-dark
    :width: 300px
    :align: left

.. image:: images/FAssMII_logo_BLACK_transparent.png
    :class: only-light
    :width: 300px
    :align: left

.. image:: images/Logo_Stiftung_Hochschullehre_neg.png
    :class: only-dark
    :width: 300px
    :align: right

.. image:: images/Logo_Stiftung_Hochschullehre_pos.png
    :class: only-light
    :width: 300px
    :align: right
