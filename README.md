# Oil Drop / Surfactant on GPU

Custom LAMMPS code and inputs for dissipative particle dynamics (DPD), with
CPU/MPI and Kokkos/CUDA implementations. The current simulation drives one
rigid sphere through fluid confined between stationary walls. Some older
input headers retain the original oil-droplet/surfactant description.

## Run on Clemson Palmetto

Clone into a shared directory that is visible from compute nodes:

```bash
git clone https://github.com/ryan1w/Oil-Drop-Surfactant-on-GPU.git
cd Oil-Drop-Surfactant-on-GPU
```

Follow [the Palmetto setup instructions](cluster/palmetto/README) to confirm
the available compiler, CUDA, MPI, and CMake modules. Then submit an A100
build and short validation run:

```bash
sbatch cluster/palmetto/smoke.sbatch a100
```

The job builds the bundled custom engine and runs 5,500 steps on the 4x
system. It requests one GPU, one MPI rank, six CPU cores, 32 GB RAM, and two
hours including compilation. Check `palmetto-smoke-JOBID.out` for
`SMOKE PASSED` and inspect the simulation log before longer runs.

Separate builds are supported for V100 and H100 nodes:

```bash
sbatch --gpus=v100:1 cluster/palmetto/smoke.sbatch v100
sbatch --gpus=h100:1 cluster/palmetto/smoke.sbatch h100
```

This checkout requires C++20, CMake >=3.22, and CUDA >=12.2. The Palmetto
scripts use CUDA 12.x to cover all three GPU types. Local script checks
passed; builds and simulations on Palmetto still need to be validated.

## Contents

| Path | Purpose |
| --- | --- |
| `lammps26/` | Complete tracked source snapshot of the modified 2026 LAMMPS engine, including bundled Kokkos |
| `lammps/` | Legacy modified 2018 source directory, retained as a reference |
| `in.dpd`, `in.dpd26` | Original and modern short simulation protocols |
| `in.prod26` | Production input with a timestep ramp and replica-dependent seeds |
| `data_sim_4x.in` | 288,442-particle starting configuration, approximately 26.5 MB |
| `cluster/palmetto/` | Portable module setup, build script, and Slurm smoke job |
| `run_gpu.sh`, `run_prod.sh` | Original workstation launchers; Palmetto users should start with the Slurm script |

`lammps26/` is included as ordinary files, so cloning does not require
submodule initialization or a separate upstream download. The source includes
the `dpd/diffcut` force, fluid/sphere and fluid/wall treatments, their Kokkos
implementations, and the fix that preserves GPU pair noise across consecutive
`run` commands. See `SOURCE_PROVENANCE.txt` for the imported source revision.

Only `data_sim_4x.in` is included from the project simulation datasets.
The 8x and 16x datasets, compiled binaries, local build directories, large
archives, and previous simulation outputs are excluded. Git LFS is not needed.

## Production status

The short cluster job checks deployment and output creation. Statistical
validation, longer equilibration, and timing on the target GPU are still
required before production. The production input currently writes restart
files after equilibration and after the driven segment; interrupted jobs do
not automatically resume. Add periodic checkpoints and a continuation input
before a campaign that could exceed its wall-time allocation.

LAMMPS license and attribution are retained in [lammps26/LICENSE](lammps26/LICENSE),
its source headers, and [lammps26/CITATION.cff](lammps26/CITATION.cff).
