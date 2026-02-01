## Dataset Preparation (`HDFieldDataset`)

We provide a custom PyTorch `Dataset` module, `HDFieldDataset`, designed to parse ensemble simulation data. Data should be prepared with relevant path arguments provided in the experiment's configuration file.

The following paths are required:

* **`data_dir`**: The path to a directory containing the individual member fields. Each file in this directory represents a single scalar/vector field from a simulation run with a specific parameter setting.
* **`cond_path`**: The path to a `.npy` file containing simulation parameters and file indices. This is a 2D array where each row corresponds to a simulation member.
    * Column 0: The index of the member field in `data_dir` (assuming files are sorted lexicographically).
    * Remaining columns: The min-max normalized parameters for that field.
* **`cond_split_path`**: The path to a `.npy` file containing a dictionary that specifies the train/test split.
    * `'cond_idx'`: An array of indices for the training members.
    * `'cond_idx_unseen'`: An array of indices for the testing members.
* **`impsmp_cond_path`** (Optional): A 1D NumPy array containing the importance score for each ensemble member. Required if using importance-based sampling.
* **`impsmp_coord_path`** (Optional): A path to a directory containing importance scores for coordinates within each field. Required if using importance-based sampling.


## Simulation Parameters and Ensemble Configuration

This directory documents the specific simulation parameters varied to generate the ensemble members for the **Nyx**, **MPAS-Ocean**, and **Cloverleaf3D** datasets. The exact parameter values corresponding to each ensemble member are archived in the `param_source` directory.

### Ensemble Splits and Temporal Scope
For all datasets, the ensemble members are ordered sequentially. The initial set constitutes the training split, while the subsequent members are reserved for testing. The dataset represents the final state of the simulation run (the target scalar field) extracted at the specific timestep listed below.

* **Nyx (Cosmology):**
    * **Train/Test Split:** First 100 members (Train) / Remaining members (Test).
    * **Temporal Duration:** Simulations were evolved for 200 timesteps.
    * **Data Scope:** The dark matter density scalar field is extracted at the final timestep ($t=200$).

* **MPAS-Ocean (Oceanography):**
    * **Train/Test Split:** First 70 members (Train) / Remaining members (Test).
    * **Temporal Duration:** Simulations were run for a duration of 15 model days.
    * **Data Scope:** The temperature scalar field is extracted at the end of Day 15.

* **Cloverleaf3D (Hydrodynamics):**
    * **Train/Test Split:** First 500 members (Train) / Remaining members (Test).
    * **Temporal Duration:** Simulations were evolved for 200 timesteps.
    * **Data Scope:** The energy scalar field is extracted at the final timestep ($t=200$).

### Official Resources
For further details regarding the underlying physics solvers and simulation environments, please refer to the official documentation and repositories:
* **Nyx:** [https://amrex-astro.github.io/Nyx/](https://amrex-astro.github.io/Nyx/)
* **MPAS-Ocean:** [https://github.com/pwolfram/MPAS-Model](https://github.com/pwolfram/MPAS-Model)
* **Cloverleaf3D:** [https://uk-mac.github.io/CloverLeaf3D/](https://uk-mac.github.io/CloverLeaf3D/)