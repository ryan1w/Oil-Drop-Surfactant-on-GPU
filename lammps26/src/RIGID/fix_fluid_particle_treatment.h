/* LAMMPS spherical solvent boundary with finite-mass rigid-body recoil.
   This file is distributed under the GNU General Public License. */
#ifdef FIX_CLASS
// clang-format off
FixStyle(fluid/particle/treatment,FixFluidParticleTreatment);
// clang-format on
#else
#ifndef LMP_FIX_FLUID_PARTICLE_TREATMENT_H
#define LMP_FIX_FLUID_PARTICLE_TREATMENT_H

#include "fix.h"
#include "fix_rigid_small.h"
#include <vector>

namespace LAMMPS_NS {

class FixFluidParticleTreatment : public Fix {
 public:
  FixFluidParticleTreatment(class LAMMPS *, int, char **);
  ~FixFluidParticleTreatment() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_integrate() override;
  void end_of_step() override;
  double compute_vector(int) override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  // Collective read-only geometry query for the separate planar wall fix.
  // wrapped = 1 returns the body COM as stored by the rigid fix (in the
  // periodic box) instead of the unwrapped COM.
  void wall_geometry(double *, double &, double &, int wrapped = 0);

  // One near-boundary solvent atom.  Exchanged between processes as raw
  // bytes, so it must stay a plain struct.
  struct Contact {
    tagint tag;
    int owner,index;
    double r[3],v[3],mass,shift[3];
  };

 protected:
  char *rigid_id;
  FixRigidSmall *rigid;
  double radius,delta,temperature,maxpenetration;
  double box_lengths[3];
  int seed,thermal;
  int kokkos_variant;   // 1 in the KOKKOS subclass: requires the /kk integrators
  double stats[20];
  double cumulative_penetrations,cumulative_treatments;
  std::vector<tagint> penetrated_tags;
  void process_boundary(int);

  // Per-atom loops over the owned atoms.  The KOKKOS subclass replaces them
  // with device kernels; collect returns only the candidate contacts, and
  // apply writes the processed contacts back.  The collective, serial,
  // tag-sorted contact processing between the two stays in process_boundary().
  virtual void scan_setup(const FixRigidSmall::FPTState &, int &, double &, double &);
  virtual void collect_contacts(int, const FixRigidSmall::FPTState &,
                                std::vector<Contact> &, int &, double &);
  virtual void apply_contacts(int, const std::vector<Contact> &);
};
}    // namespace LAMMPS_NS
#endif
#endif
