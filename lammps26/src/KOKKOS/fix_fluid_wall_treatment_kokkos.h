/* -*- c++ -*- ----------------------------------------------------------
   KOKKOS variant of fix fluid/wall/treatment (src/RIGID).
   Distributed under the GNU General Public License, version 2 or later.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(fluid/wall/treatment/kk,FixFluidWallTreatmentKokkos<LMPDeviceType>);
FixStyle(fluid/wall/treatment/kk/device,FixFluidWallTreatmentKokkos<LMPDeviceType>);
FixStyle(fluid/wall/treatment/kk/host,FixFluidWallTreatmentKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_FLUID_WALL_TREATMENT_KOKKOS_H
#define LMP_FIX_FLUID_WALL_TREATMENT_KOKKOS_H

#include "fix_fluid_wall_treatment.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

// The three per-atom loops of the CPU fix (planar reflection after the drift,
// validity scan with the solvent-wall gaps, Gaussian velocity reset in the two
// thermal layers after the final kick) run as device kernels with Kokkos
// reductions for the statistics.  The collectives, errors, statistics and
// restart data stay in the host code of the base class, in the same order.

template<class DeviceType>
class FixFluidWallTreatmentKokkos : public FixFluidWallTreatment {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixFluidWallTreatmentKokkos(class LAMMPS *, int, char **);

 protected:
  void reflect_atoms(double *, double *, int &) override;
  void scan_atoms(const double *, double, int &, double *) override;
  void thermalize_atoms(double *) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
