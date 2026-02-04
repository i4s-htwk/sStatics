======================
Installation Developer
======================

To create a development environment, you have to make sure that the
requirements from :doc:`Installation <installation>` are installed on your
system.

First clone the repository to your system and change your current working
directory inside the downloaded folder.

.. code:: console

   git clone https://github.com/i4s-htwk/sStatics.git
   cd sStatics

Then create and activate a virtual environment.

.. code:: console

   py -3.11 -m venv venv
   .\venv\Scripts\activate.bat

Finally install sStatics's development dependencies and make the package
editable.

.. code:: console

   pip install -e .[dev]
   pre-commit install

.. note::

   The `pre-commit` hooks automatically run code checks before each commit,
   helping to maintain consistent code style and catch common errors early.

=============
Documentation
=============

To generate and edit the sStatics documentation locally, you need to have
`Pandoc` installed. Pandoc is used to convert between different formats,
which is especially important for including `Jupyter` notebooks in the
Sphinx (`.rst`) documentation.

More information on installing and using Pandoc can be found here:

- https://pandoc.org/
- https://docs.readthedocs.com/platform/stable/guides/jupyter.html#background

To build the documentation and start a live preview server:

...if you are using Windows:
.. code:: console

   cd doc
   start make livehtml

...if you are using macOS:
.. code:: console

   cd doc
   make livehtml

This command launches a live server that automatically rebuilds the
documentation whenever you edit source files or notebooks, allowing you
to see changes in real time.


