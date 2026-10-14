// clang-format off
/* ----------------------------------------------------------------------
   KOKKOS variant of fix fluid/wall/treatment, 2026.
   Distributed under the GNU General Public License, version 2 or later.

   The per-atom loops of the CPU fix (RIGID/fix_fluid_wall_treatment.cpp)
   run as device kernels with Kokkos reductions:
     reflect_atoms()    specular reflection of solvent past a wall plane
     scan_atoms()       validity of solvent and wall atoms, smallest gaps,
                        no solvent inside the sphere (wrapped COM, O(1)
                        minimum image)
     thermalize_atoms() Gaussian velocity reset in the two thermal layers,
                        same stateless counter RNG as the CPU fix
   The host code of the base class does the collectives, errors, statistics
   and restart data, in the same order as the CPU fix.  Sums of the
   statistics are reduced in a different order than on the CPU.
------------------------------------------------------------------------- */

#include "fix_fluid_wall_treatment_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "fluid_treatment_kokkos.h"
#include "update.h"

#include <cfloat>

using namespace LAMMPS_NS;
using namespace FluidTreatmentKokkos;

// functors of this file; a named namespace keeps nvcc kernel names simple
namespace FWTKokkosImpl {

struct FWTReflectValue {
  double count[2], impulse[2], maxdepth[3];
  int bad;
};

struct FWTScanValue {
  double gap[2];
  int bad;
};

struct FWTThermalValue {
  double count[2], impulse[6], energy;
};

/* ----------------------------------------------------------------------
   reflect solvent that crossed a wall plane during the drift
   (cf. FixFluidWallTreatment::reflect_atoms)
------------------------------------------------------------------------- */

template<class DeviceType>
struct FWTReflectFunctor {
  typedef DeviceType device_type;
  typedef FWTReflectValue value_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_int_1d mask, type;
  typename AT::t_kkfloat_1d rmass, mass;
  int use_rmass, groupbit;
  double lo, hi, maxpenetration;

  KOKKOS_INLINE_FUNCTION
  void init(value_type &u) const {
    u.count[0] = u.count[1] = 0.0;
    u.impulse[0] = u.impulse[1] = 0.0;
    u.maxdepth[0] = u.maxdepth[1] = u.maxdepth[2] = 0.0;
    u.bad = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    for (int k = 0; k < 2; k++) {
      dst.count[k] += src.count[k];
      dst.impulse[k] += src.impulse[k];
    }
    for (int k = 0; k < 3; k++)
      if (dst.maxdepth[k] < src.maxdepth[k]) dst.maxdepth[k] = src.maxdepth[k];
    if (src.bad) dst.bad = 1;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &u) const {
    if (!(mask(i) & groupbit)) return;
    const double z = static_cast<double>(x(i,2));
    if (!finite_value(z)) {
      u.bad = 1;
      return;
    }
    const int side = (z < lo) ? 0 : ((z > hi) ? 1 : -1);
    if (side < 0) return;
    const double depth = (side == 0) ? lo - z : z - hi;
    if (depth > maxpenetration) {
      u.bad = 1;
      return;
    }
    const double m = use_rmass ? static_cast<double>(rmass(i)) : static_cast<double>(mass(type(i)));
    const double oldv = static_cast<double>(v(i,2));
    u.count[side] += 1.0;
    // impulse received by the fixed wall, opposite to the solvent change
    u.impulse[side] += 2*m*oldv;
    if (u.maxdepth[side] < depth) u.maxdepth[side] = depth;
    if (u.maxdepth[2] < 2*depth) u.maxdepth[2] = 2*depth;
    x(i,2) = (side == 0) ? lo + depth : hi - depth;
    v(i,2) = -oldv;
  }
};

/* ----------------------------------------------------------------------
   validity of all owned solvent and wall atoms, smallest solvent-wall gaps
   (cf. FixFluidWallTreatment::scan_atoms)
------------------------------------------------------------------------- */

template<class DeviceType>
struct FWTScanFunctor {
  typedef DeviceType device_type;
  typedef FWTScanValue value_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_int_1d mask, type;
  typename AT::t_tagint_1d tag;
  typename AT::t_kkfloat_1d rmass, mass;
  int use_rmass, groupbit, wallbit;
  double lo, hi;
  int has_sphere;
  double center[3], rsqmin, prd[3], half[3];
  int periodic[3];

  KOKKOS_INLINE_FUNCTION
  void init(value_type &u) const {
    u.gap[0] = u.gap[1] = DBL_MAX;
    u.bad = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    for (int k = 0; k < 2; k++)
      if (src.gap[k] < dst.gap[k]) dst.gap[k] = src.gap[k];
    if (src.bad) dst.bad = 1;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &u) const {
    const int m = mask(i);
    if (!(m & (groupbit | wallbit))) return;
    const double z = static_cast<double>(x(i,2));
    for (int d = 0; d < 3; d++)
      if (!finite_value(static_cast<double>(x(i,d))) || !finite_value(static_cast<double>(v(i,d))))
        u.bad = 1;
    if (m & wallbit) {
      if ((z > lo+1e-8) && (z < hi-1e-8)) u.bad = 1;
      for (int d = 0; d < 3; d++)
        if (Kokkos::fabs(static_cast<double>(v(i,d))) > 1e-14) u.bad = 1;
    }
    if (!(m & groupbit)) return;
    if ((z < lo) || (z > hi)) u.bad = 1;
    const double mone = use_rmass ? static_cast<double>(rmass(i)) : static_cast<double>(mass(type(i)));
    if (!(mone > 0) || !finite_value(mone) || (tag(i) <= 0)) u.bad = 1;
    if (z-lo < u.gap[0]) u.gap[0] = z-lo;
    if (hi-z < u.gap[1]) u.gap[1] = hi-z;
    if (has_sphere) {
      double dr[3];
      bool good = true;
      for (int d = 0; d < 3; d++) {
        dr[d] = static_cast<double>(x(i,d)) - center[d];
        good = good && finite_value(dr[d]);
      }
      if (!good) {
        u.bad = 1;
        return;
      }
      for (int d = 0; d < 3; d++) dr[d] = minimum_image(dr[d], prd[d], half[d], periodic[d]);
      if (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2] < rsqmin) u.bad = 1;
    }
  }
};

/* ----------------------------------------------------------------------
   Gaussian velocity reset of owned solvent in the two thermal layers
   (cf. FixFluidWallTreatment::thermalize_atoms)
------------------------------------------------------------------------- */

template<class DeviceType>
struct FWTThermalFunctor {
  typedef DeviceType device_type;
  typedef FWTThermalValue value_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_int_1d mask, type;
  typename AT::t_tagint_1d tag;
  typename AT::t_kkfloat_1d rmass, mass;
  int use_rmass, groupbit, seed;
  bigint step;
  double lo, hi, delta, temperature;

  KOKKOS_INLINE_FUNCTION
  void init(value_type &u) const {
    u.count[0] = u.count[1] = 0.0;
    for (int k = 0; k < 6; k++) u.impulse[k] = 0.0;
    u.energy = 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    dst.count[0] += src.count[0];
    dst.count[1] += src.count[1];
    for (int k = 0; k < 6; k++) dst.impulse[k] += src.impulse[k];
    dst.energy += src.energy;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &u) const {
    if (!(mask(i) & groupbit)) return;
    const double z = static_cast<double>(x(i,2));
    const int side = (z-lo < delta) ? 0 : ((hi-z < delta) ? 1 : -1);
    if (side < 0) return;
    const double m = use_rmass ? static_cast<double>(rmass(i)) : static_cast<double>(mass(type(i)));
    double g[3];
    gaussian3_wall(seed, step, tag(i), side, g);
    const double sigma = Kokkos::sqrt(temperature/m);
    u.count[side] += 1.0;
    for (int d = 0; d < 3; d++) {
      const double oldv = static_cast<double>(v(i,d)), newv = sigma*g[d];
      u.impulse[3*side+d] += m*(oldv-newv);
      u.energy += 0.5*m*(newv*newv-oldv*oldv);
      v(i,d) = newv;
    }
  }
};

}    // namespace FWTKokkosImpl

using namespace FWTKokkosImpl;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixFluidWallTreatmentKokkos<DeviceType>::FixFluidWallTreatmentKokkos(LAMMPS *lmp, int narg,
    char **arg) : FixFluidWallTreatment(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  // post_integrate reflects positions and velocities, end_of_step resets
  // velocities; masks of arrays the atom style lacks (rmass) are ignored
  datamask_read = X_MASK | V_MASK | MASK_MASK | TAG_MASK | TYPE_MASK | RMASS_MASK;
  datamask_modify = X_MASK | V_MASK;

  // the base class init() then requires the /kk integrators, and geometry()
  // asks the sphere fix for the wrapped COM
  kokkos_variant = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixFluidWallTreatmentKokkos<DeviceType>::reflect_atoms(double *local, double *maxima,
                                                           int &bad)
{
  atomKK->sync(execution_space, X_MASK | V_MASK | MASK_MASK | TYPE_MASK | RMASS_MASK);
  if (!atom->rmass) atomKK->k_mass.template sync<DeviceType>();

  FWTReflectFunctor<DeviceType> f;
  f.x = atomKK->k_x.view<DeviceType>();
  f.v = atomKK->k_v.view<DeviceType>();
  f.mask = atomKK->k_mask.view<DeviceType>();
  f.type = atomKK->k_type.view<DeviceType>();
  f.use_rmass = atom->rmass ? 1 : 0;
  if (atom->rmass) f.rmass = atomKK->k_rmass.view<DeviceType>();
  else f.mass = atomKK->k_mass.view<DeviceType>();
  f.groupbit = groupbit;
  f.lo = lo;
  f.hi = hi;
  f.maxpenetration = maxpenetration;

  FWTReflectValue result;
  Kokkos::parallel_reduce("fluid/wall/treatment/kk reflect",
                          Kokkos::RangePolicy<DeviceType>(0, atom->nlocal), f, result);
  if (result.count[0] + result.count[1] > 0)
    atomKK->modified(execution_space, X_MASK | V_MASK);

  // layout of the CPU fix: counts in 0,1; wall impulse z components in 8,11
  local[0] += result.count[0];
  local[1] += result.count[1];
  local[8] += result.impulse[0];
  local[11] += result.impulse[1];
  for (int k = 0; k < 3; k++)
    if (maxima[k] < result.maxdepth[k]) maxima[k] = result.maxdepth[k];
  if (result.bad) bad = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixFluidWallTreatmentKokkos<DeviceType>::scan_atoms(const double *center, double r,
                                                        int &bad, double *gaps)
{
  atomKK->sync(execution_space, X_MASK | V_MASK | MASK_MASK | TAG_MASK | TYPE_MASK | RMASS_MASK);
  if (!atom->rmass) atomKK->k_mass.template sync<DeviceType>();

  FWTScanFunctor<DeviceType> f;
  f.x = atomKK->k_x.view<DeviceType>();
  f.v = atomKK->k_v.view<DeviceType>();
  f.mask = atomKK->k_mask.view<DeviceType>();
  f.type = atomKK->k_type.view<DeviceType>();
  f.tag = atomKK->k_tag.view<DeviceType>();
  f.use_rmass = atom->rmass ? 1 : 0;
  if (atom->rmass) f.rmass = atomKK->k_rmass.view<DeviceType>();
  else f.mass = atomKK->k_mass.view<DeviceType>();
  f.groupbit = groupbit;
  f.wallbit = wallbit;
  f.lo = lo;
  f.hi = hi;
  f.has_sphere = sphere ? 1 : 0;
  f.rsqmin = r*r*(1-2e-12);
  for (int d = 0; d < 3; d++) {
    f.center[d] = center[d];
    f.prd[d] = domain->prd[d];
    f.half[d] = domain->prd_half[d];
    f.periodic[d] = domain->periodicity[d];
  }

  FWTScanValue result;
  Kokkos::parallel_reduce("fluid/wall/treatment/kk check",
                          Kokkos::RangePolicy<DeviceType>(0, atom->nlocal), f, result);
  for (int k = 0; k < 2; k++)
    if (result.gap[k] < gaps[k]) gaps[k] = result.gap[k];
  if (result.bad) bad = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixFluidWallTreatmentKokkos<DeviceType>::thermalize_atoms(double *local)
{
  atomKK->sync(execution_space, X_MASK | V_MASK | MASK_MASK | TAG_MASK | TYPE_MASK | RMASS_MASK);
  if (!atom->rmass) atomKK->k_mass.template sync<DeviceType>();

  FWTThermalFunctor<DeviceType> f;
  f.x = atomKK->k_x.view<DeviceType>();
  f.v = atomKK->k_v.view<DeviceType>();
  f.mask = atomKK->k_mask.view<DeviceType>();
  f.type = atomKK->k_type.view<DeviceType>();
  f.tag = atomKK->k_tag.view<DeviceType>();
  f.use_rmass = atom->rmass ? 1 : 0;
  if (atom->rmass) f.rmass = atomKK->k_rmass.view<DeviceType>();
  else f.mass = atomKK->k_mass.view<DeviceType>();
  f.groupbit = groupbit;
  f.seed = seed;
  f.step = update->ntimestep;
  f.lo = lo;
  f.hi = hi;
  f.delta = delta;
  f.temperature = temperature;

  FWTThermalValue result;
  Kokkos::parallel_reduce("fluid/wall/treatment/kk heatbath",
                          Kokkos::RangePolicy<DeviceType>(0, atom->nlocal), f, result);
  if (result.count[0] + result.count[1] > 0) atomKK->modified(execution_space, V_MASK);

  // layout of the CPU fix: counts in 0,1; momentum change in 2-7; energy in 8
  local[0] += result.count[0];
  local[1] += result.count[1];
  for (int k = 0; k < 6; k++) local[2+k] += result.impulse[k];
  local[8] += result.energy;
}

namespace LAMMPS_NS {
template class FixFluidWallTreatmentKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixFluidWallTreatmentKokkos<LMPHostType>;
#endif
}
