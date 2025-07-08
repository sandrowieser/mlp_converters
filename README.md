# mlp_converters
Provides some file converters for unconventional formats employed for several machine learned potentials such as ML_ABN or .cfg files

These codes were developed for the work described in:
Wieser, S., Zojer, E. Machine learned force-fields for an Ab-initio quality description of metal-organic frameworks. npj Comput Mater 10, 18 (2024). https://doi.org/10.1038/s41524-024-01205-w

If you use this code, please cite the paper.

## Installation

```
pip install -e .
```

## Usage

From the VASP force field format `ML_ABN`:

```
mlp_convert ML_ABN config_list.extxyz
```

For the `.cfg` format from the MLIP package implementing moment tensor potentials:

```
mlp_convert config_list.cfg config_list.extxyz C H
```

Here, it is necessary to specify the element names in order of the MLIP `.cfg` index as they are not stored inherently.

Supported input and output formats:
 - Formats supported by ASE
 - ML_AB VASP potential format
 - MLIP .cfg format

Special legacy input file formats supported by the `geometry` class with limited output types - use at your own risk:
 - Quantum Espresso `.dyn` format - requires cellconstructor
 - molsys `.mfpx` format (created by the Schmid group at the Ruhr University Bochum)

Please consult the code of `mlp_convert` for the API to obtain a list of ASE atoms objects from both of these file formats
