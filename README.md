# Oil Drop / Surfactant on GPU

Custom LAMMPS code and inputs for dissipative particle dynamics (DPD), with
CPU/MPI and Kokkos/CUDA implementations. The current simulation drives one
rigid sphere through fluid confined between stationary walls. Some older
input headers retain the original oil-droplet/surfactant description.

## Run on Clemson Palmetto

```bash
git clone https://github.com/ryan1w/Oil-Drop-Surfactant-on-GPU.git
cd Oil-Drop-Surfactant-on-GPU
# copy data_sim_8x.in and data_sim_16x.in here first (not in Git), then:
sbatch cluster/palmetto/run.sbatch 4x
sbatch cluster/palmetto/run.sbatch 8x
sbatch cluster/palmetto/run.sbatch 16x
```

Each job runs the reference protocol `in.dpd26` for one box on one GPU. The
first job on a GPU type also builds the GPU executable (20-40 min); jobs that
start meanwhile wait for it. Step-by-step instructions and notes are in
[cluster/palmetto/README](cluster/palmetto/README). The build needs C++20,
CMake >= 3.22 and CUDA 12.2 or newer (12.x for V100 support).

## Contents

| Path | Purpose |
| --- | --- |
| `lammps26/` | Complete tracked source snapshot of the modified 2026 LAMMPS engine, including bundled Kokkos |
| `lammps/` | Legacy modified 2018 source directory, retained as a reference |
| `in.dpd` | The reference script (LAMMPS 2018, CPU) |
| `in.dpd26` | The same protocol for the 2026 engine and the GPU; only `neighbor 0.3 bin` (was `multi`) and no `comm_modify mode multi` lines differ |
| `in.prod26` | Optional longer protocol for precise mobility measurements (gentler time-step ramp, fixed seeds); not used by the Palmetto scripts |
| `data_sim_4x.in` | 288,442-particle starting configuration, approximately 26.5 MB |
| `cluster/palmetto/` | Palmetto job script `run.sbatch`, module setup `setup.sh`, data checksums `data.sha256` |
| `run_gpu.sh`, `run_prod.sh` | Workstation launchers (not used on Palmetto) |

`lammps26/` is included as ordinary files, so cloning does not require
submodule initialization or a separate upstream download. The source includes
the `dpd/diffcut` force, fluid/sphere and fluid/wall treatments, their Kokkos
implementations, and the fix that preserves GPU pair noise across consecutive
`run` commands. See `SOURCE_PROVENANCE.txt` for the imported source revision.

Only `data_sim_4x.in` is included from the project simulation datasets.
The 8x and 16x datasets, compiled binaries, local build directories, large
archives, and previous simulation outputs are excluded. Git LFS is not needed.

## Comparing with the original

GPU runs are not bit-identical to the original CPU code (forces are summed in a
different order), so compare averages over several runs per box with the
reference results. With the sphere's rotation frame fixed and the pair noise
off, the GPU code reproduces the CPU code to about 1e-13 with identical event
counts in the 4x, 8x and 16x boxes. Full 4x runs (48 GPU and 16 CPU runs) also
agree statistically.

LAMMPS license and attribution are retained in [lammps26/LICENSE](lammps26/LICENSE),
its source headers, and [lammps26/CITATION.cff](lammps26/CITATION.cff).
