#!/bin/bash
# Run production replicas of in.prod26 on the GPU, one job after another
# (or SLOTS jobs at a time on the one GPU).
#
# usage: run_prod.sh CAMPAIGN_DIR JOB [JOB ...] [-- extra LAMMPS arguments]
#   JOB = BOX:REP, e.g. 4x:1 8x:2 16x:3 (BOX = 4x, 8x or 16x; REP = replica,
#         which sets every random seed, see in.prod26)
#   SLOTS=2 run_prod.sh ...   runs two jobs at a time on the GPU
# Each job runs in CAMPAIGN_DIR/BOX_rREP (log.lammps, com.dat, restart files).
# A job whose log.lammps is complete is skipped; an incomplete one is moved
# aside to BOX_rREP.incomplete.<time> and run again.  Progress goes to
# CAMPAIGN_DIR/campaign.log; the code and input versions to provenance.txt.
set -u
HERE=/home/wan6/cwork/ali/surfactant
LMP=$HERE/lammps26/build_kk/lmp
KKPKG="neigh half newton on comm device"
SLOTS=${SLOTS:-1}

[ $# -ge 2 ] || { sed -n '2,14p' "$0"; exit 1; }
camp=$(readlink -f "$1"); shift
jobs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do jobs+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
extra=("$@")

mkdir -p "$camp" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$camp/campaign.log"; }

{
  echo "date: $(date)"
  echo "binary: $LMP  md5 $(md5sum < "$LMP" | cut -d' ' -f1)"
  echo "lammps26 commit: $(git -C $HERE/lammps26 rev-parse HEAD)"
  echo "lammps26 uncommitted changes:"; git -C $HERE/lammps26 status --short | sed 's/^/  /'
  echo "in.prod26 md5: $(md5sum < $HERE/in.prod26 | cut -d' ' -f1)"
  echo "package options: $KKPKG; extra arguments: ${extra[*]:-none}"
  echo "jobs: ${jobs[*]}"
} >> "$camp/provenance.txt"

run_job() {
  local box=${1%%:*} rep=${1##*:}
  local dir="$camp/${box}_r${rep}"
  case "$box" in 4x|8x|16x) ;; *) log "skip $1: unknown box"; return 1 ;; esac
  if [ -f "$dir/log.lammps" ] && grep -q "^Total wall time" "$dir/log.lammps"; then
    log "skip ${box}_r${rep}: already complete"; return 0
  fi
  if [ -e "$dir" ]; then
    mv "$dir" "$dir.incomplete.$(date +%Y%m%d-%H%M%S)"
    log "moved incomplete ${box}_r${rep} aside"
  fi
  mkdir -p "$dir" && cd "$dir" || return 1
  ln -s "$HERE/data_sim_$box.in" data_sim.in
  cp -p "$HERE/in.prod26" .
  log "start ${box}_r${rep}"
  local t0=$SECONDS status=0
  env -u DISPLAY mpirun -np 1 "$LMP" -k on g 1 -sf kk -pk kokkos $KKPKG \
      -in in.prod26 -var rep "$rep" "${extra[@]}" -log log.lammps -screen none \
      > screen.out 2>&1 || status=$?
  log "end ${box}_r${rep}: exit $status, $(( (SECONDS - t0) / 60 )) min, $(grep -c ERROR log.lammps) ERROR, $(grep -c WARNING log.lammps) WARNING lines, dangerous builds: $(grep '^Dangerous builds' log.lammps | awk '{s+=$4} END {print s+0}')"
  return $status
}

log "campaign start: ${#jobs[@]} jobs, $SLOTS at a time"
running=0
for j in "${jobs[@]}"; do
  ( run_job "$j" ) &
  running=$((running + 1))
  if [ "$running" -ge "$SLOTS" ]; then wait -n; running=$((running - 1)); fi
done
wait
log "campaign end"
