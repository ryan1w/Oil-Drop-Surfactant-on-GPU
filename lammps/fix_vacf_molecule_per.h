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

FixStyle(vacf/molecule/per,FixVacfMoleculePer)

#else

#ifndef LMP_FIX_VACF_MOLECULE_PER_H
#define LMP_FIX_VACF_MOLECULE_PER_H

#include "stdio.h"
#include "fix.h"
#include <fstream>

namespace LAMMPS_NS {

class FixVacfMoleculePer : public Fix {
 public:
  FixVacfMoleculePer(class LAMMPS *, int, char **);
  ~FixVacfMoleculePer();
  int setmask();
  void end_of_step();

 private:
  int nmolecules;
  int idlo,idhi;
  int firstflag;

  int Nstart, Nfreq, Nlength;
  std::fstream fp;

  double *massproc,*masstotal;
  double **com,**comall,***cominit;
  int size;
  double ***vacf;
  int *molmap; // convert molecule ID to local index
  int compute_molecules_in_group(int &, int &); //compute molecule number in a group

  void compute_com();
  

};

}

#endif
#endif
