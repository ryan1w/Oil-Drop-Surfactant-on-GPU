/* -*- c++ -*- ----------------------------------------------------------
   Device helpers shared by fix fluid/particle/treatment/kk and
   fix fluid/wall/treatment/kk.  Distributed under the GNU General Public
   License.
------------------------------------------------------------------------- */

#ifndef LMP_FLUID_TREATMENT_KOKKOS_H
#define LMP_FLUID_TREATMENT_KOKKOS_H

#include "kokkos_type.h"

#include <cfloat>
#include <cstdint>

namespace LAMMPS_NS {
namespace FluidTreatmentKokkos {

/* ----------------------------------------------------------------------
   Stateless counter RNG of the CPU wall fix (anonymous namespace of
   RIGID/fix_fluid_wall_treatment.cpp), with the same keys (seed, absolute
   step, atom ID, wall side) and the same 64-bit integer arithmetic, so the
   uniform variates are bit-identical on host and device.  Only
   log/sqrt/cos/sin come from the device math library and may differ from
   the host library in the last bits.  The unsigned arithmetic intentionally
   wraps modulo 2^64.  Only the wall fix draws variates on the device: the
   sphere fix draws its few heatbath variates on the host, in the shared
   FixFluidParticleTreatment::process_boundary().
------------------------------------------------------------------------- */

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
uint64_t mix64(uint64_t x)
{
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
double uniform_open(uint64_t key, uint64_t lane)
{
  const uint64_t bits = mix64(key + UINT64_C(0x9e3779b97f4a7c15) * lane);
  return (static_cast<double>(bits >> 12) + 0.5) / 4503599627370496.0;
}

// three unit Gaussian variates from a Box-Muller pair and a half pair
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void gaussian3_from_key(uint64_t key, double *z)
{
  const double twopi = 6.283185307179586476925286766559;
  double a = Kokkos::sqrt(-2.0 * Kokkos::log(uniform_open(key, 1)));
  double theta = twopi * uniform_open(key, 2);
  z[0] = a * Kokkos::cos(theta);
  z[1] = a * Kokkos::sin(theta);
  a = Kokkos::sqrt(-2.0 * Kokkos::log(uniform_open(key, 3)));
  theta = twopi * uniform_open(key, 4);
  z[2] = a * Kokkos::cos(theta);
}

// fix fluid/wall/treatment: separate RNG namespace; keys seed, step,
// solvent atom ID and wall side (0 = lower, 1 = upper)
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void gaussian3_wall(int seed, bigint step, tagint tag, int side, double *g)
{
  uint64_t key = mix64(static_cast<uint64_t>(seed) ^ UINT64_C(0x57414c4c48454154));
  key = mix64(key ^ static_cast<uint64_t>(step));
  key = mix64(key ^ static_cast<uint64_t>(tag));
  key = mix64(key ^ static_cast<uint64_t>(side));
  gaussian3_from_key(key, g);
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
bool finite_value(double x)
{
  return (x == x) && (Kokkos::fabs(x) <= DBL_MAX);
}

/* ----------------------------------------------------------------------
   O(1) minimum image of one component of a separation in an orthogonal
   box.  The /kk fixes measure from the wrapped body COM (body xcm as stored
   by fix rigid/small, remapped into the box at every reneighboring), so
   one shift is all that normally happens; rounding handles any finite
   value without a loop.  A separation of exactly half a box length keeps
   its sign, as in the CPU while-loop.
------------------------------------------------------------------------- */

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
double minimum_image(double delta, double prd, double half, int periodic)
{
  if (periodic && (Kokkos::fabs(delta) > half)) delta -= prd * Kokkos::round(delta / prd);
  return delta;
}

}    // namespace FluidTreatmentKokkos
}    // namespace LAMMPS_NS

#endif
