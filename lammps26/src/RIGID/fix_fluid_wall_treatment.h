/* Stationary planar solvent boundaries for LAMMPS. GPL-2.0-or-later. */
#ifdef FIX_CLASS
// clang-format off
FixStyle(fluid/wall/treatment,FixFluidWallTreatment);
// clang-format on
#else
#ifndef LMP_FIX_FLUID_WALL_TREATMENT_H
#define LMP_FIX_FLUID_WALL_TREATMENT_H
#include "fix.h"
namespace LAMMPS_NS {
class FixFluidParticleTreatment;
class FixFluidWallTreatment : public Fix {
 public:
  FixFluidWallTreatment(class LAMMPS *,int,char **);
  ~FixFluidWallTreatment() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_integrate() override;
  void end_of_step() override;
  double compute_vector(int) override;
  void write_restart(FILE *) override;
  void restart(char *) override;
 protected:
  char *wall_id;
  int wallbit,seed,thermal;
  int kokkos_variant;   // 1 in the KOKKOS subclass: /kk integrators, wrapped sphere COM
  double lo,hi,delta,temperature,maxpenetration;
  double bounds[6],stats[20],cumulative[4];
  FixFluidParticleTreatment *sphere;
  void geometry(double *,double &);
  void check_atoms(int);
  // Per-atom loops over the owned atoms, replaced by device kernels in the
  // KOKKOS subclass.  They fill the local (per-process) sums, maxima and
  // flags; the collectives, errors and statistics stay in the callers.
  virtual void reflect_atoms(double *,double *,int &);
  virtual void scan_atoms(const double *,double,int &,double *);
  virtual void thermalize_atoms(double *);
};
}    // namespace LAMMPS_NS
#endif
#endif
