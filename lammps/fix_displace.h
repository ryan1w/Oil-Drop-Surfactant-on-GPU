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

FixStyle(displace,FixDisplace)

#else

#ifndef LMP_FIX_DISPLACE_H
#define LMP_FIX_DISPLACE_H

#include <stdio.h>
#include "fix.h"
#include <fstream>

namespace LAMMPS_NS {

class FixDisplace : public Fix {
 public:
  FixDisplace(class LAMMPS *, int, char **);
  ~FixDisplace();
  int setmask();
  void pre_force(int);

 private:
  int nmolecules;
  int idlo, idhi;
  int firstflag;
  int tip_type[3];

  int Nstart, Nfreq, Nlength;
  std::fstream fpdisp;

  double *massproc,*masstotal;
  double **com,**comall,***cominit;
  int size;
  double ***disp;

  void compute_com();
  int *molmap; // convert molecule ID to local index
  int compute_molecules_in_group(int &, int &); //compute molecule number in a group

};

}

#endif
#endif
