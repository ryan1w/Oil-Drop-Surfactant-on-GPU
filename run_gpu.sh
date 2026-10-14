#!/bin/bash
# Run in.dpd26 (the in.dpd protocol for LAMMPS develop 2026) on one GPU with
# the KOKKOS build of lammps26.
#
# usage: run_gpu.sh RUNDIR [4x|8x|16x|DATAFILE] [extra LAMMPS arguments ...]
#   RUNDIR    created if needed; must not contain a log.lammps yet
#   data      data_sim_4x.in (default), data_sim_8x.in, data_sim_16x.in or a path;
#             linked as RUNDIR/data_sim.in, which in.dpd26 reads
# Output in RUNDIR: log.lammps, screen.out, com.dat (fix print), in.dpd26 (copy).
#
# KOKKOS package options: pair dpd/diffcut/kk needs half neighbor lists with
# newton on; the remaining choices were the fastest ones measured on the
# RTX 4090 (see runs/kk_selfcheck/perf).
set -e
HERE=/home/wan6/cwork/ali/surfactant
LMP=$HERE/lammps26/build_kk/lmp
KKPKG="neigh half newton on comm device"

[ $# -ge 1 ] || { sed -n '2,13p' "$0"; exit 1; }
rundir=$1; shift
data=${1:-4x}; [ $# -ge 1 ] && shift
case "$data" in
  4x|8x|16x) data=$HERE/data_sim_$data.in ;;
esac
data=$(readlink -f "$data")
[ -f "$data" ] || { echo "data file $data not found"; exit 1; }
[ -x "$LMP" ] || { echo "KOKKOS binary $LMP not found"; exit 1; }

mkdir -p "$rundir"
cd "$rundir"
[ -e log.lammps ] && { echo "$rundir already has a log.lammps; use a new run directory"; exit 1; }
ln -sfn "$data" data_sim.in
cp -p $HERE/in.dpd26 .

echo "run_gpu.sh: $(date) data $data" > screen.out
status=0
env -u DISPLAY mpirun -np 1 "$LMP" -k on g 1 -sf kk -pk kokkos $KKPKG \
    -in in.dpd26 -log log.lammps "$@" >> screen.out 2>&1 || status=$?
echo "exit status $status; $(grep -c ERROR log.lammps) ERROR lines in $rundir/log.lammps"
exit $status
