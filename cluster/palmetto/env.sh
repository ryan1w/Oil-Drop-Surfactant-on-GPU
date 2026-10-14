#!/bin/bash
# Source from a Palmetto login shell. Module versions can be overridden with
# PALMETTO_GCC_MODULE, PALMETTO_CUDA_MODULE, PALMETTO_MPI_MODULE and
# PALMETTO_CMAKE_MODULE. See README for the documented starting versions.

PALMETTO_PROJECT=${PALMETTO_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}
PALMETTO_PROJECT=$(cd -- "$PALMETTO_PROJECT" && pwd -P)
PALMETTO_BUILD_ROOT=${PALMETTO_BUILD_ROOT:-$PALMETTO_PROJECT/build/palmetto}
if [[ $PALMETTO_BUILD_ROOT != /* ]]; then
    PALMETTO_BUILD_ROOT=$PALMETTO_PROJECT/$PALMETTO_BUILD_ROOT
fi

palmetto_gpu_arch() {
    case "$1" in
        v100) printf '%s\n' VOLTA70 ;;
        a100) printf '%s\n' AMPERE80 ;;
        h100) printf '%s\n' HOPPER90 ;;
        *) echo "Supported GPU targets: v100, a100, h100" >&2; return 1 ;;
    esac
}

palmetto_load_modules() {
    if ! type module >/dev/null 2>&1; then
        echo "The module command is unavailable. Use a Palmetto login shell." >&2
        return 1
    fi
    module purge || return
    module load "${PALMETTO_GCC_MODULE:-gcc/12.3.0}" || return
    module load "${PALMETTO_CUDA_MODULE:-cuda/12.3.0}" || return
    # CUDA must be loaded before the CUDA-aware OpenMPI module is selected.
    module load "${PALMETTO_MPI_MODULE:-openmpi/5.0.1}" || return
    if [[ -n ${PALMETTO_CMAKE_MODULE:-} ]]; then
        module load "$PALMETTO_CMAKE_MODULE" || return
    fi
    export OMP_NUM_THREADS=1
}
