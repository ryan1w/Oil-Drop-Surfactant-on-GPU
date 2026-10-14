#!/bin/bash
# Run on an allocated compute node: bash cluster/palmetto/build.sh a100
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

[[ $# -eq 1 ]] || { echo "Usage: $0 v100|a100|h100" >&2; exit 2; }
gpu=$1
arch=$(palmetto_gpu_arch "$gpu")
[[ -n ${SLURM_JOB_ID:-} ]] || {
    echo "Build inside a Slurm allocation; see cluster/palmetto/README." >&2
    exit 2
}
palmetto_load_modules
for program in cmake nvcc gcc g++ mpicxx sha256sum; do
    command -v "$program" >/dev/null || { echo "Missing command: $program" >&2; exit 1; }
done

cuda_version=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p')
if [[ ! $cuda_version =~ ^12\.([0-9]+)$ ]] || (( ${BASH_REMATCH[1]:-0} < 2 )); then
    echo "Use CUDA 12.2 or newer within the 12.x series (found $cuda_version)." >&2
    echo "This Kokkos needs CUDA >=12.2; CUDA 13 cannot compile for V100." >&2
    exit 1
fi
cmake_version=$(cmake --version | sed -n '1s/^cmake version //p')
if [[ ! $cmake_version =~ ^([0-9]+)\.([0-9]+) ]] ||
   (( ${BASH_REMATCH[1]:-0} < 3 || (${BASH_REMATCH[1]:-0} == 3 && ${BASH_REMATCH[2]:-0} < 22) )); then
    echo "CMake >=3.22 is required. Set PALMETTO_CMAKE_MODULE to an available module." >&2
    exit 1
fi

source_dir=$PALMETTO_PROJECT/lammps26
[[ -f $source_dir/src/KOKKOS/pair_dpd_diffcut_kokkos.cpp ]] || {
    echo "Missing custom LAMMPS source in $source_dir" >&2; exit 1;
}
build_dir=$PALMETTO_BUILD_ROOT/$gpu
mkdir -p "$build_dir"
build_dir=$(cd -- "$build_dir" && pwd -P)
export NVCC_WRAPPER_DEFAULT_COMPILER
NVCC_WRAPPER_DEFAULT_COMPILER=$(command -v g++)
jobs=${PALMETTO_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-1}}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo "Invalid build job count: $jobs" >&2; exit 2; }
(( jobs <= ${SLURM_CPUS_PER_TASK:-1} )) || {
    echo "Build parallelism exceeds allocated CPUs." >&2; exit 2;
}

# Avoid silently reusing a CMake cache from a different compiler/MPI stack.
{
    module -t list 2>&1
    command -v cmake nvcc gcc g++ mpicxx
    cmake --version
    nvcc --version
    g++ --version
    mpicxx --showme
} > "$build_dir/toolchain.requested.txt"
if [[ -f $build_dir/toolchain.txt ]] &&
   ! cmp -s "$build_dir/toolchain.txt" "$build_dir/toolchain.requested.txt"; then
    echo "Toolchain changed. Choose a new PALMETTO_BUILD_ROOT for this build." >&2
    exit 1
fi
cp "$build_dir/toolchain.requested.txt" "$build_dir/toolchain.txt"
cp "${BASH_SOURCE[0]}" "$build_dir/build-script.sh"
# Hash the actual files, including changes that have not been committed.
(
    cd -- "$source_dir"
    find src cmake lib -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$build_dir/source.sha256"

cmake -S "$source_dir/cmake" -B "$build_dir" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_C_COMPILER="$(command -v gcc)" \
    -D CMAKE_CXX_COMPILER="$source_dir/lib/kokkos/bin/nvcc_wrapper" \
    -D MPI_CXX_COMPILER="$(command -v mpicxx)" \
    -D CMAKE_C_FLAGS= -D CMAKE_CXX_FLAGS= \
    -D BUILD_MPI=ON -D BUILD_OMP=OFF -D BUILD_SHARED_LIBS=OFF \
    -D PKG_DPD-BASIC=ON -D PKG_MOLECULE=ON -D PKG_RIGID=ON -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=OFF -D Kokkos_ARCH_NATIVE=OFF \
    -D "Kokkos_ARCH_$arch=ON" -D KOKKOS_PREC=double \
    -D EXTERNAL_KOKKOS=OFF -D DOWNLOAD_KOKKOS=OFF \
    2>&1 | tee "$build_dir/configure.log"
cmake --build "$build_dir" --parallel "$jobs" 2>&1 | tee "$build_dir/build.log"
sha256sum "$build_dir/lmp" > "$build_dir/binary.sha256"
printf 'Built %s for %s (%s)\n' "$build_dir/lmp" "$gpu" "$arch"
