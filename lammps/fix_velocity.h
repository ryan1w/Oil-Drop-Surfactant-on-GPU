/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(velocity,FixVelocity)

#else

#ifndef LMP_FIX_VELOCITY_H
#define LMP_FIX_VELOCITY_H

#include "stdio.h"
#include "fix.h"
#include <fstream>

namespace LAMMPS_NS {

class FixVelocity : public Fix {
 public:
  FixVelocity(class LAMMPS *, int, char **);
  ~FixVelocity();
  int setmask();
  void end_of_step();

 private:
  int nmolecules;
  int idlo,idhi;
  int firstflag;
  int tip_type[3];

  int Nstart, Nfreq, Nlength;
  std::fstream fpvel;

  double *massproc,*masstotal;
  double **com_v,**comall_v,***cominit,**com_r,**comall_r;
  int size;
  double ***velocity;
  int *molmap; // convert molecule ID to local index
  int compute_molecules_in_group(int &, int &); //compute molecule number in a group

  void compute_comv_project();

};

}

#endif
#endif
