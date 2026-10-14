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

/* ----------------------------------------------------------------------
   Contributing authors: Salahudin Sheikh &  Zhen Li (Clemson U)
   This fix is designed for computing the center of mass of the molecules
------------------------------------------------------------------------- */
#include "atom.h"
#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "fix_com_salah.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "domain.h"
#include "comm.h"
#include "math.h"
#include <mpi.h>
#include <fstream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */

FixComSalah::FixComSalah(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix_com command");

  Nstart = force->inumeric(FLERR,arg[3]);
  Nfreq = force->inumeric(FLERR,arg[4]);
  Nlength = force->inumeric(FLERR,arg[5]);
  R0 = force->inumeric(FLERR,arg[6]);
 
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  memory->create(com,4,"fix/com/salah:com");
  memory->create(comall,4,"fix/com/salah:comall");

  size = int(Nlength/Nfreq);
  memory->create(comvel,size,8,"fix/com/salah:comvel");
  memory->create(comvelall,size,8,"fix/com/salah:comvelall");
  
  for(int i = 0; i < size; i++)
  for(int j = 0; j < 8; j++) {
    comvel[i][j] = 0.0;
	comvelall[i][j] = 0.0;
  }

  //=======================================================================
  if(comm->me == 0) {
    fpcomvel.open("comvel.dat",ios::trunc | ios::out);
    fpcomvel << "Variables = \"time\", \"vx1\", \"vy1\", \"vz1\", \"vx2\", \"vy2\", \"vz2\"\n";
  }
  //=======================================================================
}

/* ---------------------------------------------------------------------- */

FixComSalah::~FixComSalah()
{
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(comvel);
  memory->destroy(comvelall);
}

/* ---------------------------------------------------------------------- */

int FixComSalah::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixComSalah::end_of_step()
{
	bigint ntimestep = update->ntimestep;
	double dsq;
	int step;
	int *mask = atom->mask;
	int *type = atom->type;
	double **v = atom->v;
	double **x = atom->x;
	int nlocal = atom->nlocal;

	if((ntimestep > Nstart) && (ntimestep < Nlength)) {
		step = (ntimestep - Nstart) % Nlength;
		if(step % Nfreq == 0) {
			compute_com();
			int index = step / Nfreq;
			for(int j = 0; j < 8; j++) {
				comvel[index][j] = 0.0;
				comvelall[index][j] = 0.0;
			}
			for (int i = 0; i < nlocal; i++){
				if(mask[i] & groupbit){
					dsq = (x[i][0]-comall[0])*(x[i][0]-comall[0]) + (x[i][1]-comall[1])*(x[i][1]-comall[1]);
					if(x[i][2]<comall[2] & dsq <= R0*R0){
						comvel[index][0] += v[i][0];
						comvel[index][1] += v[i][1];
						comvel[index][2] += v[i][2];
						comvel[index][3] += 1;
					}
				}
				else if(type[i] == 6 || type[i] == 7 || type[i] ==8){
					dsq = (x[i][0]-comall[0])*(x[i][0]-comall[0]) + (x[i][1]-comall[1])*(x[i][1]-comall[1]);
					if(x[i][2]<comall[2] & dsq <= R0*R0){
						comvel[index][4] += v[i][0];
						comvel[index][5] += v[i][1];
						comvel[index][6] += v[i][2];
						comvel[index][7] += 1;
					}		
				}
			}
			MPI_Allreduce(&comvel[index][0],&comvelall[index][0],8,MPI_DOUBLE,MPI_SUM,world);				
		}
	}
	 

  if(ntimestep == update->laststep && comm->me == 0) {
    for(int i = 0; i < size; i++) 
      fpcomvel << i*Nfreq*update->dt <<"  "<< comvelall[i][0] <<"  "<< comvelall[i][1] <<"  "<< comvelall[i][2] <<" "<< comvelall[i][4] <<" "<< comvelall[i][5] <<" "<< comvelall[i][6] <<endl;
  }
}


// compute current COM positions
void FixComSalah::compute_com()
{
  double massone, mass_inv;
  double unwrap[3];

  for (int i = 0; i < 3; i++)
      com[i] = 0.0;

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  tagint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      domain->unmap(x[i],image[i],unwrap);
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      com[0] += unwrap[0] * massone;
      com[1] += unwrap[1] * massone;
      com[2] += unwrap[2] * massone;
    }
  }
	MPI_Allreduce(&com[0],&comall[0],3,MPI_DOUBLE,MPI_SUM,world);

    mass_inv = 1.0 / nlocal;
    comall[0] *= mass_inv;
    comall[1] *= mass_inv;
    comall[2] *= mass_inv;
}


