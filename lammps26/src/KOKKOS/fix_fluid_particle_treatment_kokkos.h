/* -*- c++ -*- ----------------------------------------------------------
   KOKKOS variant of fix fluid/particle/treatment (src/RIGID).
   Distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(fluid/particle/treatment/kk,FixFluidParticleTreatmentKokkos<LMPDeviceType>);
FixStyle(fluid/particle/treatment/kk/device,FixFluidParticleTreatmentKokkos<LMPDeviceType>);
FixStyle(fluid/particle/treatment/kk/host,FixFluidParticleTreatmentKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_FLUID_PARTICLE_TREATMENT_KOKKOS_H
#define LMP_FIX_FLUID_PARTICLE_TREATMENT_KOKKOS_H

#include "fix_fluid_particle_treatment.h"
#include "kokkos_type.h"

#include <vector>

namespace LAMMPS_NS {

// one candidate contact found by the device scan (see collect_contacts())
struct FPTDeviceContact {
  tagint tag;
  int index;
  double mass;
  double r[3], v[3];
};

// one processed contact owned by this process, written back on the device:
// value = position shift (phase 0) or new velocity (phase 1)
struct FPTDeviceApply {
  tagint tag;
  int index;
  int remap;
  double value[3];
};

// The per-step work of the sphere fix is split into a scan of all owned
// solvent (device kernel), a serial, collective, tag-sorted processing of the
// few contacts it finds (host code of the base class, shared with the CPU
// fix) and a write-back of those contacts (small device kernel).  The rigid
// body is read and updated through the FixRigidSmallKokkos hooks, which work
// on its device body array.  Distances are measured from the wrapped body COM
// with an O(1) minimum image; the CPU fix uses the unwrapped COM.

template<class DeviceType>
class FixFluidParticleTreatmentKokkos : public FixFluidParticleTreatment {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixFluidParticleTreatmentKokkos(class LAMMPS *, int, char **);

 protected:
  void scan_setup(const FixRigidSmall::FPTState &, int &, double &, double &) override;
  void collect_contacts(int, const FixRigidSmall::FPTState &, std::vector<Contact> &,
                        int &, double &) override;
  void apply_contacts(int, const std::vector<Contact> &) override;

  typedef Kokkos::DualView<FPTDeviceContact *, Kokkos::LayoutRight, DeviceType> tdual_contact_1d;
  typedef Kokkos::DualView<FPTDeviceApply *, Kokkos::LayoutRight, DeviceType> tdual_apply_1d;
  typedef Kokkos::DualView<tagint *, Kokkos::LayoutRight, DeviceType> tdual_tag_1d;

  // buffers, grown on demand; host and device copies are moved explicitly
  tdual_contact_1d k_contacts;    // compacted candidate contacts
  tdual_apply_1d k_apply;         // owned contacts to write back
  tdual_tag_1d k_pentags;         // sorted IDs penetrated in phase 0 of this step
  Kokkos::View<int, DeviceType> d_slot;    // next free slot in k_contacts
};

}    // namespace LAMMPS_NS

#endif
#endif
