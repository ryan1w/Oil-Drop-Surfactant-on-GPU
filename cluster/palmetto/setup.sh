# Sourced by run.sbatch.  The only place to edit if Palmetto's module names
# differ (check once with: module avail gcc cuda openmpi cmake).

module purge
module load gcc/12.3.0
module load cuda/12.3.0         # CUDA 12.2 or newer is required; CUDA 13 cannot build for V100
module load openmpi/5.0.1
export OMP_NUM_THREADS=1

# Kokkos GPU architecture of the GPU in this job, from its name
gpu_arch() {
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    case "$name" in
        *H100*|*H200*)        echo HOPPER90 ;;
        *A100*)               echo AMPERE80 ;;
        *A40*)                echo AMPERE86 ;;
        *L40*|*"RTX 40"*)     echo ADA89 ;;
        *V100*)               echo VOLTA70 ;;
        *P100*)               echo PASCAL60 ;;
        *) echo "Unknown GPU '$name': add it to gpu_arch() in cluster/palmetto/setup.sh" >&2
           return 1 ;;
    esac
}

# Build lammps26/ into build/<arch>/lmp unless it is there already.  Only one
# job builds for a GPU type; jobs that start meanwhile wait and then use it.
build_lammps() {
    local arch=$1 src=$PWD/lammps26 bdir=$PWD/build/$1 lock=$PWD/build/$1.lock owner
    mkdir -p build
    until ln -s "${SLURM_JOB_ID:-$$}" "$lock" 2>/dev/null; do
        owner=$(readlink "$lock")
        if [ -n "$owner" ] && ! squeue -h -o %i -j "$owner" 2>/dev/null | grep -q .; then
            rm -f "$lock"           # the job that held the lock is gone
            continue
        fi
        echo "Waiting for job $owner to finish building for $arch ..."
        sleep $((45 + RANDOM % 30))
    done
    trap "rm -f '$lock'" EXIT
    if [ ! -x "$bdir/lmp" ]; then
        cmake_ok() { cmake --version 2>/dev/null | awk 'NR==1 {split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=22))}'; }
        cmake_ok || module load cmake 2>/dev/null || true
        cmake_ok || { echo "CMake 3.22 or newer is needed: add a cmake module to cluster/palmetto/setup.sh"; return 1; }
        echo "Building LAMMPS for $arch in $bdir (first job for this GPU type, 20-40 min)"
        cmake -S "$src/cmake" -B "$bdir" -D CMAKE_BUILD_TYPE=Release \
              -D CMAKE_CXX_COMPILER="$src/lib/kokkos/bin/nvcc_wrapper" -D CMAKE_CXX_STANDARD=20 \
              -D BUILD_MPI=on -D PKG_MOLECULE=on -D PKG_RIGID=on -D PKG_DPD-BASIC=on \
              -D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on -D "Kokkos_ARCH_$arch=on" > "$PWD/build/$arch-cmake.log" 2>&1 ||
            { echo "CMake failed, see build/$arch-cmake.log"; return 1; }
        cmake --build "$bdir" --parallel "${SLURM_CPUS_PER_TASK:-8}" > "$PWD/build/$arch-build.log" 2>&1 ||
            { echo "Compilation failed, see build/$arch-build.log"; return 1; }
        local style
        for style in dpd/diffcut/kk fluid/particle/treatment/kk fluid/wall/treatment/kk print/kk; do
            "$bdir/lmp" -h | grep -q -- "$style" || { echo "Style $style is missing from $bdir/lmp"; return 1; }
        done
        echo "Built $bdir/lmp"
    fi
    rm -f "$lock"
    trap - EXIT
}
