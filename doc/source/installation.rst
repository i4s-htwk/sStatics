.. _label-installation:

============
Installation
============

Make sure you have the following installed on your system:

* `Python3 <https://www.python.org/downloads>`_, version 3.11 or higher
* `Git <https://git-scm.com/downloads>`_

.. tip::
   We recommend to install sStatics into a virtual environment. This avoids
   conflicts with other Python packages on your system.

First, change to a directory where you want to store the virtual environment,
then create and activate it:

.. code:: console

   python -m venv venv
   venv\Scripts\activate

Download and install the current version of sStatics:

.. code:: console

   python -m pip install git+https://github.com/i4s-htwk/sStatics.git

To see that everything works fine, you can run tests on the sample exercises:

.. code:: console

   python -m sstatics test

You should see an output similar to the following:

.. code:: console

   ----------------------------------------------------------------------
   Ran 63 tests in 0.137s

   OK
