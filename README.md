# AGI-SoulOS

This repository now includes an executable reference implementation of the
**Grand Soul Operating System v10.0** cognitive architecture described in the
original specification.  The implementation lives in `soul_os_v10.py` and can
be executed directly:

```bash
python soul_os_v10.py
```

Running the script will simulate five cognitive cycles, printing the internal
metrics (free energy, resonance, belief averages, etc.) for each iteration.

> **Note:** The simulation depends on NumPy.  Install it with `pip install numpy`
> before executing the script if it is not already available in your
> environment.