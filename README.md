
<div align="center">
  <img src="doc/source/images/sStatics_Logo.png" alt="logo" width="200">
</div>

`sStatics` is a Python package for the analysis of planar truss structures 
using the displacement method. `sStatics` can be used to calculate 
displacements,  internal forces, and support reactions, and to visualise 
structural behavior under applied loads. The package supports second-order 
analysis and influence lines for advanced structural assessment.

<div align="center">
  <img src="doc/source/images/tutorial1.png" alt="logo" width="400">
</div>

## Getting Started

This section explains how to install `sStatics` as a Python package so you can 
use it in your own projects.

Instructions for installing the project for development purposes are 
provided in the full documentation [sStatics Documentation](https://sStatics.readthedocs.io/).

### Prerequisites

- [Python 3 (version >= 3.11)](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

Open a terminal, navigate to your desired installation directory, and run 
the following commands.

### On Windows (CMD)

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install git+https://github.com/i4s-htwk/sStatics.git
```

### On macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install git+https://github.com/i4s-htwk/sStatics.git
```

## Documentation

`sStatics` is fully documented including a user walkthrough, examples and 
an API guide. The documentation can be found at [sStatics Documentation](https://sStatics.readthedocs.io/).


## License
This project is licensed under the GNU Affero General Public License v3.0 
(AGPL-3.0).


## Support

Found a bug 🐛, or have a feature request ✨, raise an issue on the GitHub 
[issue tracker](https://github.com/i4s-htwk/sStatics/issues). Alternatively 
you can get support on the [discussions page](https://github.com/orgs/i4s-htwk/discussions).

## Disclaimer
`sStatics` is an open-source engineering tool originally developed within a 
research project at HTWK Leipzig and is now maintained and further 
developed by the i4s institute. While care has been taken to implement the 
underlying structural mechanics and numerical methods correctly—including 
the displacement method, second-order effects, and influence line 
calculations — it remains the user's responsibility to verify and assess all 
results independently. The authors and contributors do not assume liability 
for any incorrect use or interpretation of the software. Refer to the 
license for details regarding permitted use and limitations.

## Acknowledgements

This project is developed at the HTWK Leipzig University of Applied 
Sciences and funded by the "Stiftung Innovation in der Hochschullehre".

<div align="center">
  <img src="doc/source/images/FAssMII_logo_WHITE_transparent.png" 
alt="logo" width="200" style="margin-right: 20px;">
  <img src="doc/source/images/Logo_Stiftung_Hochschullehre_neg.png" 
alt="logo" width="200">
</div>
