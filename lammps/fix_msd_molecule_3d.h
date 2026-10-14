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

FixStyle(msd/molecule/3d,FixMsdMolecule3d)

#else

#ifndef LMP_FIX_MSD_MOLECULE_3D_H
#define LMP_FIX_MSD_MOLECULE_3D_H

#include <stdio.h>
#include "fix.h"
#include <fstream>

namespace LAMMPS_NS {

class FixMsdMolecule3d : public Fix {
 public:
  FixMsdMolecule3d(class LAMMPS *, int, char **);
  ~FixMsdMolecule3d();
  int setmask();
  void pre_force(int);

 private:
  int nmolecules;
  int idlo, idhi;
  int firstflag;
  int tip_type[3];
  int direction;
  double value;

  int Nstart, Nfreq, Nlength;
  std::fstream fpmsd;

  double *massproc,*masstotal;
  double **com,**comall,***cominit;
  int size;
  double ***msd;

  void compute_com();
  int *molmap; // convert molecule ID to local index
  int compute_molecules_in_group(int &, int &); //compute molecule number in a group
  char *filename; //User defined filename for the output file
  int namelength;

};

}

#endif
#endif
