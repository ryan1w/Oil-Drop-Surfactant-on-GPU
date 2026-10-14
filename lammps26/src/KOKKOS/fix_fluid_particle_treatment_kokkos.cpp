// clang-format off
/* ----------------------------------------------------------------------
   KOKKOS variant of fix fluid/particle/treatment, 2026.
   Distributed under the GNU General Public License.

   The algorithm, its order of operations, statistics, restart data and
   error conditions are those of the CPU fix (RIGID/fix_fluid_particle_
   treatment.cpp), whose process_boundary() is shared.  Only the loops over
   the owned atoms run here, as device kernels:
     scan_setup()       distances of solvent and sphere atoms at setup
     collect_contacts() validity flags, minimum (projected) distance and the
                        compacted list of candidate contacts (inside the
                        sphere in phase 0; in the thermal shell or penetrated
                        in phase 0 of this step, in phase 1)
     apply_contacts()   write-back of the processed owned contacts: position
                        shift with image-flag remap, or new velocity
   The few contacts per step travel between host and device; per-atom arrays
   never do.  The rigid body is read and updated through the
   FixRigidSmallKokkos overrides of fpt_get_state()/fpt_set_velocities().
   Distances are measured from the WRAPPED body COM with an O(1) minimum
   image (the CPU fix uses the unwrapped COM and a loop to stay bit-identical
   with the 2018 code).
------------------------------------------------------------------------- */

#include "fix_fluid_particle_treatment_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "domain_kokkos.h"
#include "error.h"
#include "fluid_treatment_kokkos.h"
#include "kokkos_few.h"

#include <cfloat>
#include <cstring>
#include <utility>

using namespace LAMMPS_NS;
using namespace FluidTreatmentKokkos;

// functors of this file; a named namespace keeps nvcc kernel names simple
namespace FPTKokkosImpl {

// reduction value of the setup scan
struct FPTSetupValue {
  double maxbead, minfluid;
  int invalid;
};

// reduction value of the contact scan
struct FPTCollectValue {
  double minradius;
  int invalid, count;
};

// geometry shared by the scans: wrapped body COM and periodic box
struct FPTGeometry {
  double xcm[3], prd[3], half[3];
  int periodic[3];
};

FPTGeometry make_geometry(const Domain *domain, const double *xcm)
{
  FPTGeometry g;
  for (int d = 0; d < 3; d++) {
    g.xcm[d] = xcm[d];
    g.prd[d] = domain->prd[d];
    g.half[d] = domain->prd_half[d];
    g.periodic[d] = domain->periodicity[d];
  }
  return g;
}

/* ----------------------------------------------------------------------
   setup: largest distance of a sphere atom and smallest distance of a
   solvent atom from the COM (cf. FixFluidParticleTreatment::scan_setup)
------------------------------------------------------------------------- */

template<class DeviceType>
struct FPTSetupFunctor {
  typedef DeviceType device_type;
  typedef FPTSetupValue value_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_int_1d mask;
  int fluidbit, bodybit;
  FPTGeometry geo;

  KOKKOS_INLINE_FUNCTION
  void init(value_type &u) const { u.maxbead = 0.0; u.minfluid = DBL_MAX; u.invalid = 0; }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    if (dst.maxbead < src.maxbead) dst.maxbead = src.maxbead;
    if (src.minfluid < dst.minfluid) dst.minfluid = src.minfluid;
    if (src.invalid) dst.invalid = 1;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &u) const {
    const int m = mask(i);
    if (!(m & (fluidbit | bodybit))) return;
    double r[3];
    for (int d = 0; d < 3; d++) r[d] = static_cast<double>(x(i,d)) - geo.xcm[d];
    if (!finite_value(r[0]) || !finite_value(r[1]) || !finite_value(r[2])) {
      u.invalid = 1;
      return;
    }
    for (int d = 0; d < 3; d++)
      r[d] = minimum_image(r[d], geo.prd[d], geo.half[d], geo.periodic[d]);
    const double dist = Kokkos::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (!finite_value(dist)) u.invalid = 1;
    if ((m & bodybit) && (u.maxbead < dist)) u.maxbead = dist;
    if ((m & fluidbit) && (dist < u.minfluid)) u.minfluid = dist;
  }
};

/* ----------------------------------------------------------------------
   per step: validity, smallest (projected) distance and the candidate
   contacts, compacted into "out" through an atomic slot counter.  The order
   of the compacted contacts does not matter: the base class gathers and
   sorts all contacts by atom ID before processing them, and atom IDs are
   unique.  If there are more candidates than slots, the count is still
   exact and the caller grows the buffer and runs the scan again.
   (cf. FixFluidParticleTreatment::collect_contacts)
------------------------------------------------------------------------- */

template<class DeviceType>
struct FPTCollectFunctor {
  typedef DeviceType device_type;
  typedef FPTCollectValue value_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_int_1d mask, type;
  typename AT::t_tagint_1d tag;
  typename AT::t_kkfloat_1d rmass, mass;
  int use_rmass, groupbit, thermal_phase;
  FPTGeometry geo;
  double radius, shell_radius;
  typename Kokkos::DualView<tagint *, Kokkos::LayoutRight, DeviceType>::t_dev pentags;
  int npentags;
  typename Kokkos::DualView<FPTDeviceContact *, Kokkos::LayoutRight, DeviceType>::t_dev out;
  int capacity;
  Kokkos::View<int, DeviceType> slot;

  KOKKOS_INLINE_FUNCTION
  void init(value_type &u) const { u.minradius = DBL_MAX; u.invalid = 0; u.count = 0; }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    if (src.minradius < dst.minradius) dst.minradius = src.minradius;
    if (src.invalid) dst.invalid = 1;
    dst.count += src.count;
  }

  // binary search in the sorted penetrated IDs
  KOKKOS_INLINE_FUNCTION
  bool penetrated(const tagint itag) const {
    int lo = 0, hi = npentags;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (pentags(mid) < itag) lo = mid + 1;
      else hi = mid;
    }
    return (lo < npentags) && (pentags(lo) == itag);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &u) const {
    if (!(mask(i) & groupbit)) return;
    const tagint itag = tag(i);
    const double m = use_rmass ? static_cast<double>(rmass(i)) : static_cast<double>(mass(type(i)));
    double r[3], vel[3];
    for (int d = 0; d < 3; d++) {
      r[d] = static_cast<double>(x(i,d)) - geo.xcm[d];
      vel[d] = static_cast<double>(v(i,d));
      if (!finite_value(r[d]) || !finite_value(vel[d])) u.invalid = 1;
    }
    if (!finite_value(r[0]) || !finite_value(r[1]) || !finite_value(r[2])) return;
    for (int d = 0; d < 3; d++)
      r[d] = minimum_image(r[d], geo.prd[d], geo.half[d], geo.periodic[d]);
    if (!finite_value(m) || !(m > 0.0) || (itag <= 0)) u.invalid = 1;
    const double dist = Kokkos::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (!finite_value(dist)) u.invalid = 1;

    // the thermal phase uses current full-step velocities and actual positions
    const double finaldist = (!thermal_phase && (dist < radius)) ? 2*radius - dist : dist;
    if (finaldist < u.minradius) u.minradius = finaldist;

    bool candidate;
    if (thermal_phase) {
      if (dist < radius*(1-1e-12)) u.invalid = 1;
      // a deeply reflected atom can lie outside the shell; it still receives
      // one thermal boundary collision during this step
      candidate = (dist < shell_radius) || ((npentags > 0) && penetrated(itag));
    } else candidate = (dist < radius);
    if (!candidate) return;

    u.count++;
    const int k = Kokkos::atomic_fetch_add(&slot(), 1);
    if (k < capacity) {
      FPTDeviceContact c;
      c.tag = itag;
      c.index = i;
      c.mass = m;
      for (int d = 0; d < 3; d++) {
        c.r[d] = r[d];
        c.v[d] = vel[d];
      }
      out(k) = c;
    }
  }
};

/* ----------------------------------------------------------------------
   write back the processed owned contacts; counts contacts whose local
   index no longer holds the same atom (cf. FixFluidParticleTreatment::
   apply_contacts).  The remap mirrors Domain::remap() for an orthogonal box.
------------------------------------------------------------------------- */

template<class DeviceType>
struct FPTApplyFunctor {
  typedef DeviceType device_type;
  typedef int value_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_imageint_1d image;
  typename AT::t_tagint_1d tag;
  typename Kokkos::DualView<FPTDeviceApply *, Kokkos::LayoutRight, DeviceType>::t_dev in;
  int thermal_phase, nlocal;
  Few<double,3> lo, hi, period;
  int xperiodic, yperiodic, zperiodic;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k, int &mismatch) const {
    const FPTDeviceApply a = in(k);
    const int i = a.index;
    if ((i < 0) || (i >= nlocal) || (tag(i) != a.tag)) {
      mismatch++;
      return;
    }
    if (thermal_phase) {
      for (int d = 0; d < 3; d++) v(i,d) = a.value[d];
      return;
    }
    for (int d = 0; d < 3; d++) x(i,d) += a.value[d];
    // keep unwrapped trajectories: update the image flags when the corrected
    // position crosses a periodic face
    if (a.remap) {
      Few<double,3> coord;
      for (int d = 0; d < 3; d++) coord[d] = static_cast<double>(x(i,d));
      imageint img = image(i);
      coord = DomainKokkos::remap(lo, hi, period, xperiodic, yperiodic, zperiodic, coord, img);
      for (int d = 0; d < 3; d++) x(i,d) = coord[d];
      image(i) = img;
    }
  }
};

}    // namespace FPTKokkosImpl

using namespace FPTKokkosImpl;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixFluidParticleTreatmentKokkos<DeviceType>::FixFluidParticleTreatmentKokkos(LAMMPS *lmp,
    int narg, char **arg) : FixFluidParticleTreatment(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  // post_integrate moves atoms and their image flags, end_of_step sets
  // velocities; masks of arrays the atom style lacks (rmass) are ignored
  datamask_read = X_MASK | V_MASK | MASK_MASK | TAG_MASK | TYPE_MASK | IMAGE_MASK | RMASS_MASK;
  datamask_modify = X_MASK | V_MASK | IMAGE_MASK;

  // the base class init() then requires the /kk integrators
  kokkos_variant = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixFluidParticleTreatmentKokkos<DeviceType>::scan_setup(const FixRigidSmall::FPTState &state,
    int &invalid, double &maxbead, double &minfluid)
{
  atomKK->sync(execution_space, X_MASK | MASK_MASK);

  FPTSetupFunctor<DeviceType> f;
  f.x = atomKK->k_x.view<DeviceType>();
  f.mask = atomKK->k_mask.view<DeviceType>();
  f.fluidbit = groupbit;
  f.bodybit = rigid->groupbit;
  f.geo = make_geometry(domain, state.xcm_wrapped);

  FPTSetupValue result;
  Kokkos::parallel_reduce("fluid/particle/treatment/kk setup scan",
                          Kokkos::RangePolicy<DeviceType>(0, atom->nlocal), f, result);
  if (result.invalid) invalid = 1;
  if (maxbead < result.maxbead) maxbead = result.maxbead;
  if (result.minfluid < minfluid) minfluid = result.minfluid;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixFluidParticleTreatmentKokkos<DeviceType>::collect_contacts(int thermal_phase,
    const FixRigidSmall::FPTState &state, std::vector<Contact> &local, int &invalid,
    double &minradius)
{
  atomKK->sync(execution_space, X_MASK | V_MASK | MASK_MASK | TAG_MASK | TYPE_MASK | RMASS_MASK);
  if (!atom->rmass) atomKK->k_mass.template sync<DeviceType>();

  // IDs penetrated in phase 0 of this step (sorted by the base class); there
  // are none on almost every step

  int npen = 0;
  if (thermal_phase && !penetrated_tags.empty()) {
    npen = static_cast<int>(penetrated_tags.size());
    if (k_pentags.extent_int(0) < npen)
      k_pentags = tdual_tag_1d("fluid/particle/treatment/kk:pentags", npen);
    for (int k = 0; k < npen; k++) k_pentags.view_host()(k) = penetrated_tags[k];
    const auto range = std::make_pair(0, npen);
    Kokkos::deep_copy(Kokkos::subview(k_pentags.template view<DeviceType>(), range),
                      Kokkos::subview(k_pentags.view_host(), range));
  }

  if (!d_slot.data()) d_slot = Kokkos::View<int, DeviceType>("fluid/particle/treatment/kk:slot");
  if (k_contacts.extent_int(0) == 0)
    k_contacts = tdual_contact_1d("fluid/particle/treatment/kk:contacts", 64);

  FPTCollectFunctor<DeviceType> f;
  f.x = atomKK->k_x.view<DeviceType>();
  f.v = atomKK->k_v.view<DeviceType>();
  f.mask = atomKK->k_mask.view<DeviceType>();
  f.type = atomKK->k_type.view<DeviceType>();
  f.tag = atomKK->k_tag.view<DeviceType>();
  f.use_rmass = atom->rmass ? 1 : 0;
  if (atom->rmass) f.rmass = atomKK->k_rmass.view<DeviceType>();
  else f.mass = atomKK->k_mass.view<DeviceType>();
  f.groupbit = groupbit;
  f.thermal_phase = thermal_phase;
  f.geo = make_geometry(domain, state.xcm_wrapped);
  f.radius = radius;
  f.shell_radius = radius + delta;
  f.pentags = k_pentags.template view<DeviceType>();
  f.npentags = npen;
  f.slot = d_slot;

  FPTCollectValue result;
  while (true) {
    Kokkos::deep_copy(d_slot, 0);
    f.out = k_contacts.template view<DeviceType>();
    f.capacity = k_contacts.extent_int(0);
    Kokkos::parallel_reduce("fluid/particle/treatment/kk contact scan",
                            Kokkos::RangePolicy<DeviceType>(0, atom->nlocal), f, result);
    if (result.count <= f.capacity) break;
    k_contacts = tdual_contact_1d("fluid/particle/treatment/kk:contacts",
                                  result.count + result.count/2 + 64);
  }

  if (result.invalid) invalid = 1;
  if (result.minradius < minradius) minradius = result.minradius;

  if (result.count > 0) {
    const auto range = std::make_pair(0, result.count);
    auto h_contacts = Kokkos::subview(k_contacts.view_host(), range);
    Kokkos::deep_copy(h_contacts, Kokkos::subview(k_contacts.template view<DeviceType>(), range));
    for (int k = 0; k < result.count; k++) {
      const FPTDeviceContact &dc = h_contacts(k);
      Contact c;
      std::memset(&c, 0, sizeof(c));
      c.tag = dc.tag;
      c.owner = comm->me;
      c.index = dc.index;
      c.mass = dc.mass;
      for (int d = 0; d < 3; d++) {
        c.r[d] = dc.r[d];
        c.v[d] = dc.v[d];
      }
      local.push_back(c);
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixFluidParticleTreatmentKokkos<DeviceType>::apply_contacts(int thermal_phase,
    const std::vector<Contact> &contacts)
{
  int n = 0;
  for (const auto &c : contacts)
    if (c.owner == comm->me) n++;
  if (n == 0) return;

  if (k_apply.extent_int(0) < n)
    k_apply = tdual_apply_1d("fluid/particle/treatment/kk:apply", n + n/2 + 16);
  auto h_apply = k_apply.view_host();
  int k = 0;
  for (const auto &c : contacts) {
    if (c.owner != comm->me) continue;
    FPTDeviceApply a;
    a.tag = c.tag;
    a.index = c.index;
    for (int d = 0; d < 3; d++) a.value[d] = thermal_phase ? c.v[d] : c.shift[d];
    const double shiftsq = c.shift[0]*c.shift[0] + c.shift[1]*c.shift[1] + c.shift[2]*c.shift[2];
    a.remap = (!thermal_phase && (shiftsq > 0)) ? 1 : 0;
    h_apply(k++) = a;
  }
  const auto range = std::make_pair(0, n);
  Kokkos::deep_copy(Kokkos::subview(k_apply.template view<DeviceType>(), range),
                    Kokkos::subview(h_apply, range));

  const uint64_t changed = thermal_phase ? V_MASK : (X_MASK | IMAGE_MASK);
  atomKK->sync(execution_space, changed | TAG_MASK);

  FPTApplyFunctor<DeviceType> f;
  f.x = atomKK->k_x.view<DeviceType>();
  f.v = atomKK->k_v.view<DeviceType>();
  f.image = atomKK->k_image.view<DeviceType>();
  f.tag = atomKK->k_tag.view<DeviceType>();
  f.in = k_apply.template view<DeviceType>();
  f.thermal_phase = thermal_phase;
  f.nlocal = atom->nlocal;
  f.lo = Few<double,3>(domain->boxlo);
  f.hi = Few<double,3>(domain->boxhi);
  f.period = Few<double,3>(domain->prd);
  f.xperiodic = domain->xperiodic;
  f.yperiodic = domain->yperiodic;
  f.zperiodic = domain->zperiodic;

  int mismatch = 0;
  Kokkos::parallel_reduce("fluid/particle/treatment/kk apply",
                          Kokkos::RangePolicy<DeviceType>(0, n), f, mismatch);
  atomKK->modified(execution_space, changed);
  if (mismatch)
    error->one(FLERR, "Contact ownership changed during fluid/particle/treatment");
}

namespace LAMMPS_NS {
template class FixFluidParticleTreatmentKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixFluidParticleTreatmentKokkos<LMPHostType>;
#endif
}
