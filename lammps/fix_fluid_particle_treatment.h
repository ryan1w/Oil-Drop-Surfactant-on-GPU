/* LAMMPS spherical solvent boundary with finite-mass rigid-body recoil.
   This file is distributed under the GNU General Public License. */
#ifdef FIX_CLASS
FixStyle(fluid/particle/treatment,FixFluidParticleTreatment)
#else
#ifndef LMP_FIX_FLUID_PARTICLE_TREATMENT_H
#define LMP_FIX_FLUID_PARTICLE_TREATMENT_H

#include "fix.h"
#include <vector>

namespace LAMMPS_NS {
class FixRigidSmall;

class FixFluidParticleTreatment : public Fix {
 public:
  FixFluidParticleTreatment(class LAMMPS *, int, char **);
  ~FixFluidParticleTreatment();
  int setmask();
  void init();
  void setup(int);
  void post_integrate();
  void end_of_step();
  double compute_vector(int);
  void write_restart(FILE *);
  void restart(char *);
  // Collective read-only geometry query for the separate planar wall fix.
  void wall_geometry(double *, double &, double &);

 private:
  char *rigid_id;
  FixRigidSmall *rigid;
  double radius,delta,temperature,maxpenetration;
  double box_lengths[3];
  int seed,thermal;
  double stats[20];
  double cumulative_penetrations,cumulative_treatments;
  std::vector<tagint> penetrated_tags;
  void process_boundary(int);
};
}
#endif
#endif
