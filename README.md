# AGI-SoulOS

This repository now includes an executable reference implementation of the
**Grand Soul Operating System v10.0** cognitive architecture described in the
original specification. The implementation lives in `soul_os_v10.py`.

## Requirements

* Python 3.10+
* [NumPy](https://numpy.org/) for numerical operations

Install the dependency with:

```bash
pip install numpy
```

## Running the simulation

Execute the cognitive-cycle simulator directly from the repository root:

```bash
python soul_os_v10.py
```

By default the program simulates five cognitive cycles and prints the
intermediate metrics (free energy, resonance, belief averages, etc.) for each
iteration.
