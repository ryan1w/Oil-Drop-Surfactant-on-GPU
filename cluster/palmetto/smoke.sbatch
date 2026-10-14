#!/bin/bash -l
#SBATCH --job-name=dpd-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --output=palmetto-smoke-%j.out
#SBATCH --error=palmetto-smoke-%j.err

# Submit from the project root. Override BOTH the Slurm GPU type and argument:
# sbatch --gpus=v100:1 cluster/palmetto/smoke.sbatch v100
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_SUBMIT_DIR:-} ]] || {
    echo "Submit with sbatch from the project root." >&2; exit 2;
}
[[ ${SLURM_NTASKS:-1} == 1 && ${SLURM_JOB_NUM_NODES:-1} == 1 ]] || {
    echo "This smoke job requires one node and one MPI rank." >&2; exit 2;
}
export PALMETTO_PROJECT=${PALMETTO_PROJECT:-$SLURM_SUBMIT_DIR}
[[ -f $PALMETTO_PROJECT/cluster/palmetto/env.sh ]] || {
    echo "PALMETTO_PROJECT must identify the transferred project root." >&2; exit 2;
}
source "$PALMETTO_PROJECT/cluster/palmetto/env.sh"
[[ $# -le 1 ]] || { echo "Usage: sbatch smoke.sbatch [v100|a100|h100]" >&2; exit 2; }
gpu=${1:-a100}
arch=$(palmetto_gpu_arch "$gpu")
palmetto_load_modules
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || {
    echo "No GPU is assigned to this job." >&2; exit 1;
}
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
if ! grep -qi -- "$gpu" <<< "$gpu_names"; then
    echo "Requested build target $gpu does not match this node's GPUs: $gpu_names" >&2
    exit 1
fi

box=${PALMETTO_BOX:-4x}
case "$box" in 4x|8x|16x) ;; *) echo "PALMETTO_BOX must be 4x, 8x or 16x" >&2; exit 2 ;; esac
data=$PALMETTO_PROJECT/data_sim_$box.in
[[ -s $data && -s $PALMETTO_PROJECT/in.prod26 ]] || {
    echo "Missing input or data file: $data" >&2; exit 1;
}
bash "$PALMETTO_PROJECT/cluster/palmetto/build.sh" "$gpu"
build_dir=$PALMETTO_BUILD_ROOT/$gpu
lmp=$build_dir/lmp

# Job IDs give every attempt its own directory, including failed attempts.
run_root=${PALMETTO_RUN_ROOT:-$PALMETTO_PROJECT/runs/palmetto}
[[ $run_root == /* ]] || run_root=$PALMETTO_PROJECT/$run_root
rundir=$run_root/smoke_${gpu}_${box}_${SLURM_JOB_ID}
mkdir -p -- "$(dirname -- "$rundir")"
mkdir -- "$rundir"
cd -- "$rundir"
cp "$PALMETTO_PROJECT/in.prod26" .
ln -s "$data" data_sim.in
cp "$build_dir/toolchain.txt" "$build_dir/source.sha256" "$build_dir/binary.sha256" .
{
    date -u
    hostname
    printf 'Slurm job: %s\nGPU target: %s (%s)\n' "$SLURM_JOB_ID" "$gpu" "$arch"
    module -t list 2>&1
    nvidia-smi
    sha256sum in.prod26 "$data"
} > provenance.txt

# Use the project's production input with shorter stages: 5500 total steps.
# This exercises the timestep ramp, both boundaries, drive, COM and restarts.
# A one-rank job also works with MPI stacks that do not support GPU buffers.
status=0
srun --ntasks=1 "$lmp" -k on g 1 -sf kk \
    -pk kokkos neigh half newton on comm device gpu/aware off \
    -in in.prod26 -var rep 1 -var ndt1 1000 -var nramp 500 \
    -var neq 2000 -var ndrive 1000 -var nthermo 100 -var ncom 10 \
    -log log.lammps -screen screen.out || status=$?
if (( status != 0 )); then
    echo "LAMMPS failed with exit status $status; outputs: $rundir" >&2
    exit "$status"
fi
for output in log.lammps com.dat restart.equil restart.drive; do
    [[ -s $output ]] || { echo "Missing output: $output" >&2; exit 1; }
done
grep -q '^Total wall time:' log.lammps
if grep -Eiq '^ERROR|(^|[[:space:]])[-+]?(nan|inf(inity)?)([[:space:]]|$)' log.lammps com.dat; then
    echo "Errors or non-finite values in $rundir" >&2; exit 1
fi
printf 'SMOKE PASSED: %s\n' "$rundir"
echo 'This confirms build/startup/output compatibility. Statistical validation is still required.'
