#!/bin/bash -l
#SBATCH --job-name=dpd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=dpd-%j.out
#
# Run the reference protocol in.dpd26 (= in.dpd, see the README) for one box
# on one GPU.  Submit from the repository root, one job per run:
#   sbatch cluster/palmetto/run.sbatch 4x
#   sbatch cluster/palmetto/run.sbatch 8x
#   sbatch cluster/palmetto/run.sbatch 16x
#   sbatch --gpus=h100:1 cluster/palmetto/run.sbatch 16x     # other GPU types: v100, l40s, a40
# The first job on a GPU type also builds LAMMPS (20-40 min, into build/<ARCH>/);
# jobs that start meanwhile wait for that build and then use it.
# Output: runs/<box>_<jobid>/ with log.lammps, com.dat and run_info.txt; this
# job's dpd-<jobid>.out ends with a short summary.

set -e
box=${1:?usage: sbatch cluster/palmetto/run.sbatch 4x|8x|16x}
case "$box" in 4x|8x|16x) ;; *) echo "The box must be 4x, 8x or 16x, not '$box'"; exit 1 ;; esac
cd "$SLURM_SUBMIT_DIR"
[ -f cluster/palmetto/setup.sh ] || { echo "Submit from the repository root (the directory with in.dpd26)"; exit 1; }
source cluster/palmetto/setup.sh

# the data file must be the reference one (8x and 16x are copied in by hand)
grep " data_sim_$box.in\$" cluster/palmetto/data.sha256 | sha256sum --check --quiet - ||
    { echo "data_sim_$box.in is missing or differs from the reference (see cluster/palmetto/README)"; exit 1; }

arch=$(gpu_arch)
lmp=$PWD/build/$arch/lmp
[ -x "$lmp" ] || build_lammps "$arch"

rundir=$PWD/runs/${box}_$SLURM_JOB_ID
mkdir -p "$rundir"
cd "$rundir"
ln -sf "$SLURM_SUBMIT_DIR/data_sim_$box.in" data_sim.in
cp "$SLURM_SUBMIT_DIR/in.dpd26" .
{
    echo "job $SLURM_JOB_ID on $(hostname), $(date)"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "box $box, executable $lmp ($arch)"
    git -C "$SLURM_SUBMIT_DIR" log -1 --format='code: commit %H, %cd' 2>/dev/null || true
    sha256sum "$lmp" in.dpd26 "$SLURM_SUBMIT_DIR/data_sim_$box.in"
} > run_info.txt

status=0
srun "$lmp" -k on g 1 -sf kk -pk kokkos neigh half newton on comm device \
     -in in.dpd26 -log log.lammps || status=$?

set +e
echo "---- summary: $box on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1), exit status $status"
awk '/^Loop time/ {printf "run %d: %d steps, %.2f ms/step\n", ++n, $9, 1000*$4/$9}' log.lammps
grep -E '^(Dangerous builds|ERROR|WARNING)' log.lammps
echo "com.dat rows: $(grep -vsc '^#' com.dat)   output: $rundir"
exit $status
