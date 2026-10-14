/* Stationary planar solvent boundaries for LAMMPS. GPL-2.0-or-later. */
#ifdef FIX_CLASS
FixStyle(fluid/wall/treatment,FixFluidWallTreatment)
#else
#ifndef LMP_FIX_FLUID_WALL_TREATMENT_H
#define LMP_FIX_FLUID_WALL_TREATMENT_H
#include "fix.h"
namespace LAMMPS_NS {
class FixFluidParticleTreatment;
class FixFluidWallTreatment : public Fix {
 public:
  FixFluidWallTreatment(class LAMMPS *,int,char **);
  ~FixFluidWallTreatment();
  int setmask();
  void init();
  void setup(int);
  void post_integrate();
  void end_of_step();
  double compute_vector(int);
  void write_restart(FILE *);
  void restart(char *);
 private:
  char *wall_id;
  int wallbit,seed,thermal;
  double lo,hi,delta,temperature,maxpenetration;
  double bounds[6],stats[20],cumulative[4];
  FixFluidParticleTreatment *sphere;
  void geometry(double *,double &);
  void check_atoms(int);
};
}
#endif
#endif
