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
   Contributing author: Zhen Li (Brown U)
   This fix is designed for computing the velocity of molecules with respect to rotational local coordinates a,b, and c
------------------------------------------------------------------------- */
#include "atom.h"
#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "fix_velocity.h"
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

FixVelocity::FixVelocity(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix_velocity command");

  Nstart = force->inumeric(FLERR,arg[3]);
  Nfreq = force->inumeric(FLERR,arg[4]);
  Nlength = force->inumeric(FLERR,arg[5]);
  tip_type[0] = force->inumeric(FLERR,arg[6]);
  tip_type[1] = force->inumeric(FLERR,arg[7]);
  tip_type[2] = force->inumeric(FLERR,arg[8]);

  if (atom->molecular == 0)
	error->all(FLERR,"fix_velocity requires molecular atom style");
 
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  molmap = NULL;
  nmolecules = compute_molecules_in_group(idlo,idhi);

  memory->create(massproc,nmolecules,"fix_velocity:massproc");
  memory->create(masstotal,nmolecules,"fix_velocity:masstotal");
  memory->create(com_v,nmolecules,3,"fix_velocity:com");
  memory->create(comall_v,nmolecules,3,"fix_velocity:comall");
  memory->create(com_r,nmolecules,12,"fix_velocity:com_r"); //carry com and three unit vectors along a, b, c
  memory->create(comall_r,nmolecules,12,"fix_velocity:comall_r");

  size = int(Nlength/Nfreq);
  memory->create(cominit,size,nmolecules,3,"fix_velocity:cominit"); // need projected velocity va, vb and vc
  memory->create(velocity,nmolecules,size,4,"fix_velocity:velocity"); // va, vb, and vc
  for(int k = 0; k < nmolecules; k++) {
	for(int i = 0; i < size; i++) {
		for (int j = 0; j < 4; j++) {
			velocity[k][i][j] = 0.0;
		}
    }
  }
  
// compute masstotal for each molecule
  int imol;
  double massone;

  for (int i = 0; i < nmolecules; i++) massproc[i] = 0.0;
  for (int i = 0; i < nlocal; i++)
	if (mask[i] & groupbit) {
		if (rmass) massone = rmass[i];
		else massone = mass[type[i]];
		imol = molecule[i];
		if (molmap) imol = molmap[imol-idlo];
		else imol--;
		massproc[imol] += massone;
    }
    
  MPI_Allreduce(massproc,masstotal,nmolecules,MPI_DOUBLE,MPI_SUM,world);

// compute initial COM velocity for each molecule
  firstflag = 1;
  compute_comv_project();

  for (int i = 0; i < nmolecules; i++)
	for (int j = 0; j < 3; j++) 
	cominit[0][i][j] = comall_v[i][j];
  
  firstflag = 0;

  //=======================================================================
  if(comm->me == 0) {
	fpvel.open("velocity",ios::trunc | ios::out);
    fpvel << "Variables = \"time\", \"v_a\"i, \"v_b\", \"v_c\"\n";
  }
//=======================================================================
}

/* ---------------------------------------------------------------------- */

FixVelocity::~FixVelocity()
{
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com_v);
  memory->destroy(comall_v);
  memory->destroy(cominit);
  memory->destroy(velocity);
  memory->destroy(molmap);
  memory->destroy(com_r);
  memory->destroy(comall_r);
}

/* ---------------------------------------------------------------------- */

int FixVelocity::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}


/* ---------------------------------------------------------------------- */

void FixVelocity::end_of_step()
{
	bigint ntimestep = update->ntimestep;
	int step;
	if((ntimestep > Nstart) && (ntimestep < Nlength)) {
		step = (ntimestep - Nstart) % Nlength;
		if(step % Nfreq == 0) {
			compute_comv_project();
			if (firstflag) error->all(FLERR,"fix_velocity error. cominit has not been set yet.");
			int index = step / Nfreq;
			for (int i = 0; i < nmolecules; i++)
			for (int j = 0; j < 3; j++)
			if(comm->me == 0) {
				for (int i = 0; i < nmolecules; i++) {
					velocity[i][index][0] = comall_v[i][0];
					velocity[i][index][1] = comall_v[i][1];
					velocity[i][index][2] = comall_v[i][2];
				}		
			}
		}
	}
	if(ntimestep == update->laststep && comm->me == 0) {
		for(int k = 0; k < nmolecules; k++) {
			fpvel << "zone i ="<< size << ",f=point" << endl;
			for(int i = 0; i < size; i++) 
			fpvel << i*Nfreq*update->dt <<"  "<< velocity[k][i][0] <<"  "<< velocity[k][i][1] <<"  "<< velocity[k][i][2] <<"  "<< nmolecules <<endl;
		}
	}
}	
// compute current COM velocity projected to a, b, c directions
void FixVelocity::compute_comv_project()
{
  int imol;
  double massone, mass_inv;
  double unwrap[3],temp_comv[3];
  double dx,dy,dz,rinv;

  for (int i = 0; i < nmolecules; i++){
    com_v[i][0] = com_v[i][1] = com_v[i][2] = 0;	  
    for (int j = 0; j < 12; j++)
      com_r[i][j] = 0.0;
  }

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int *type = atom->type;
  tagint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      imol = molecule[i];
      if (molmap) imol = molmap[imol-idlo];
      else imol--;
      domain->unmap(x[i],image[i],unwrap);
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      com_r[imol][0] += unwrap[0] * massone;
      com_r[imol][1] += unwrap[1] * massone;
      com_r[imol][2] += unwrap[2] * massone;

      com_v[imol][0] += v[i][0] * massone;
      com_v[imol][1] += v[i][1] * massone;
      com_v[imol][2] += v[i][2] * massone;

      if(type[i] == tip_type[0]){
        com_r[imol][3] += unwrap[0];
	      com_r[imol][4] += unwrap[1];
	      com_r[imol][5] += unwrap[2];
      }else if(type[i] == tip_type[1]){
        com_r[imol][6] += unwrap[0];
        com_r[imol][7] += unwrap[1];
        com_r[imol][8] += unwrap[2];
      }else if(type[i] == tip_type[2]){
        com_r[imol][9] += unwrap[0];
        com_r[imol][10] += unwrap[1];
        com_r[imol][11] += unwrap[2];
      }
  }
  
  MPI_Allreduce(&com_v[0][0],&comall_v[0][0],3*nmolecules,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&com_r[0][0],&comall_r[0][0],12*nmolecules,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nmolecules; i++) {
    mass_inv = 1.0 / masstotal[i];
    comall_r[i][0] *= mass_inv;
    comall_r[i][1] *= mass_inv;
    comall_r[i][2] *= mass_inv;

    comall_v[i][0] *= mass_inv;
    comall_v[i][1] *= mass_inv;
    comall_v[i][2] *= mass_inv;

    dx = comall_r[i][3] - comall_r[i][0];
    dy = comall_r[i][4] - comall_r[i][1];
    dz = comall_r[i][5] - comall_r[i][2];
    rinv = 1.0/sqrt(dx*dx + dy*dy + dz*dz);
    comall_r[i][3] = dx*rinv;
    comall_r[i][4] = dy*rinv;
    comall_r[i][5] = dz*rinv;

    dx = comall_r[i][6] - comall_r[i][0];
    dy = comall_r[i][7] - comall_r[i][1];
    dz = comall_r[i][8] - comall_r[i][2];
    rinv = 1.0/sqrt(dx*dx + dy*dy + dz*dz);
    comall_r[i][6] = dx*rinv;
    comall_r[i][7] = dy*rinv;
    comall_r[i][8] = dz*rinv;

    dx = comall_r[i][9] - comall_r[i][0];
    dy = comall_r[i][10] - comall_r[i][1];
    dz = comall_r[i][11] - comall_r[i][2];
    rinv = 1.0/sqrt(dx*dx + dy*dy + dz*dz);
    comall_r[i][9] = dx*rinv;
    comall_r[i][10] = dy*rinv;
    comall_r[i][11] = dz*rinv;
  }

  for (int i = 0; i < nmolecules; i++) {
    temp_comv[0] = comall_v[i][0];
    temp_comv[1] = comall_v[i][1];
    temp_comv[2] = comall_v[i][2];
    comall_v[i][0] = temp_comv[0]*comall_r[i][3] + temp_comv[1]*comall_r[i][4] + temp_comv[2]*comall_r[i][5];
    comall_v[i][1] = temp_comv[0]*comall_r[i][6] + temp_comv[1]*comall_r[i][7] + temp_comv[2]*comall_r[i][8];
    comall_v[i][2] = temp_comv[0]*comall_r[i][9] + temp_comv[1]*comall_r[i][10] + temp_comv[2]*comall_r[i][11];
  }
}

/* ----------------------------------------------------------------------
   identify molecule IDs with atoms in group
   warn if any atom in group has molecule ID = 0
   warn if any molecule has only some atoms in group
   return Ncount = # of molecules with atoms in group
   set molmap to NULL if molecule IDs include all in range from 1 to Ncount
   else: molecule IDs range from idlo to idhi
         set molmap to vector of length idhi-idlo+1
         molmap[id-idlo] = index from 0 to Ncount-1
         return idlo and idhi
------------------------------------------------------------------------- */

int FixVelocity::compute_molecules_in_group(int &idlo, int &idhi)
{
  int i;

  memory->destroy(molmap);
  molmap = NULL;

  // find lo/hi molecule ID for any atom in group
  // warn if atom in group has ID = 0

  int *molecule = atom->molecule;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int lo = 2000000000;
  int hi = 0;
  int flag = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (molecule[i] == 0) error->all(FLERR,"Atom with molecule ID = 0 included in fix_velocity::compute_molecules_in_group");
      lo = MIN(lo,molecule[i]);
      hi = MAX(hi,molecule[i]);
    }

  MPI_Allreduce(&lo,&idlo,1,MPI_INT,MPI_MIN,world);
  MPI_Allreduce(&hi,&idhi,1,MPI_INT,MPI_MAX,world);

  // molmap = vector of length nlen
  // set to 1 for IDs that appear in group across all procs, else 0

  int nlen = idhi-idlo+1;
  memory->create(molmap,nlen,"fix_velocity:molmap");
  for (i = 0; i < nlen; i++) molmap[i] = 0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      molmap[molecule[i]-idlo] = 1;

  int *molmapall;
  memory->create(molmapall,nlen,"fix_velocity:molmapall");
  MPI_Allreduce(molmap,molmapall,nlen,MPI_INT,MPI_MAX,world);

  // nmolecules = # of non-zero IDs in molmap
  // molmap[i] = index of molecule, skipping molecules not in group with -1

  int nmolecules = 0;
  for (i = 0; i < nlen; i++)
    if (molmapall[i]) molmap[i] = nmolecules++;
    else molmap[i] = -1;
  memory->destroy(molmapall);

  // warn if any molecule has some atoms in group and some not in group

  flag = 0;
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) continue;
    if (molecule[i] < idlo || molecule[i] > idhi) continue;
    if (molmap[molecule[i]-idlo] >= 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall && comm->me == 0)
    error->warning(FLERR,"One or more compute molecules has atoms not in group");

  // if molmap simply stores 1 to Nmolecules, then free it

  if (idlo == 1 && idhi == nmolecules && nlen == nmolecules) {
    memory->destroy(molmap);
    molmap = NULL;
  }
  return nmolecules;
}
