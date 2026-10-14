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
   Contributing author: Zhen Li (Clemson U)
   This fix is designed for computing the displacemnet of the molecules with respect to rotational coordinates a, b, and c.
------------------------------------------------------------------------- */
#include "atom.h"
#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "fix_displace.h"
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

FixDisplace::FixDisplace(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix_displace command");

  Nstart = force->inumeric(FLERR,arg[3]);
  Nfreq = force->inumeric(FLERR,arg[4]);
  Nlength = force->inumeric(FLERR,arg[5]);
  tip_type[0] = force->inumeric(FLERR,arg[6]);
  tip_type[1] = force->inumeric(FLERR,arg[7]);
  tip_type[2] = force->inumeric(FLERR,arg[8]);

  if (atom->molecular == 0)
    error->all(FLERR,"fix_displace requires molecular atom style");
 
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  molmap = NULL;
  nmolecules = compute_molecules_in_group(idlo,idhi);

  memory->create(massproc,nmolecules,"fix_displace:massproc");
  memory->create(masstotal,nmolecules,"fix_displace:masstotal");
  memory->create(com,nmolecules,12,"fix_displace:com");
  memory->create(comall,nmolecules,12,"fix_displace:comall");

  size = int(Nlength/Nfreq);
  memory->create(cominit,size,nmolecules,12,"fix_displace:cominit");
  memory->create(disp,nmolecules,size,4,"fix_displace:disp");
  
  for(int k = 0; k < nmolecules; k++)
  for(int i = 0; i < size; i++)
  for(int j = 0; j < 4; j++) {
    disp[k][i][j] = 0.0;
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

  // compute initial COM for each molecule
  firstflag = 1;
  compute_com();

  for (int i = 0; i < nmolecules; i++)
    for (int j = 0; j < 12; j++) 
      cominit[0][i][j] = comall[i][j];
  
  firstflag = 0;

  //=======================================================================
  if(comm->me == 0) {
    fpdisp.open("displace",ios::trunc | ios::out);
    fpdisp << "Variables = \"time\", \"disp_a\", \"disp_b\", \"disp_c\"\n";
  }
  //=======================================================================
}

/* ---------------------------------------------------------------------- */

FixDisplace::~FixDisplace()
{
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(cominit);
  memory->destroy(disp);
  memory->destroy(molmap);
}

/* ---------------------------------------------------------------------- */

int FixDisplace::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDisplace::pre_force(int vflag)
{
	bigint ntimestep = update->ntimestep;
	int step;
	int *mask = atom->mask;
	int *type = atom->type;
	int *molecule = atom->molecule;
	int nlocal = atom->nlocal;
	if((ntimestep > Nstart) && (ntimestep < Nlength)) {
		step = (ntimestep - Nstart) % Nlength;
    if(step % Nfreq == 0) {
			double dx, dy, dz;
			double dr_a, dr_b, dr_c;
      compute_com();
      if (firstflag) error->all(FLERR,"fix_displace error. cominit has not been set yet.");	
			int index = step / Nfreq;
			for (int i = 0; i < nmolecules; i++)
			for (int j = 0; j < 12; j++)
			cominit[index][i][j] = comall[i][j];
      if(comm->me == 0) {
				for(int i = 0; i < nmolecules; i++) {
				dx = comall[i][0] - cominit[index-1][i][0];
				dy = comall[i][1] - cominit[index-1][i][1];
				dz = comall[i][2] - cominit[index-1][i][2];
				dr_a = dx*cominit[index-1][i][3] + dy*cominit[index-1][i][4] + dz*cominit[index-1][i][5];
				dr_b = dx*cominit[index-1][i][6] + dy*cominit[index-1][i][7] + dz*cominit[index-1][i][8];
				dr_c = dx*cominit[index-1][i][9] + dy*cominit[index-1][i][10] + dz*cominit[index-1][i][11];
        disp[i][index][0] = dr_a;
				disp[i][index][1] = dr_b;
				disp[i][index][2] = dr_c;   
        }
			}
		}
	} 

  if(ntimestep == update->laststep && comm->me == 0) {
    for(int k = 0; k < nmolecules; k++) {
      fpdisp << "zone i ="<< size << ",f=point" << endl;
      for(int i = 0; i < size; i++) 
      fpdisp << i*Nfreq*update->dt <<"  "<< disp[k][i][0] <<"  "<< disp[k][i][1] <<"  "<< disp[k][i][2] <<"  "<< nmolecules <<endl;
      //fpdisp << i*Nfreq*update->dt <<"  "<< disp[k][i][0] <<"  "<< disp[k][i][1] <<"  "<< disp[k][i][2] <<endl;
    }
  }
}
// compute current COM positions
void FixDisplace::compute_com()
{
  int imol;
  double massone, mass_inv;
  double unwrap[3];
  double dx,dy,dz,rinv;

  for (int i = 0; i < nmolecules; i++)
    for (int j = 0; j < 12; j++)
      com[i][j] = 0.0;

  double **x = atom->x;
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
      com[imol][0] += unwrap[0] * massone;
      com[imol][1] += unwrap[1] * massone;
      com[imol][2] += unwrap[2] * massone;

      if(type[i] == tip_type[0]){
        com[imol][3] += unwrap[0];
		com[imol][4] += unwrap[1];
		com[imol][5] += unwrap[2];
      }else if(type[i] == tip_type[1]){
        com[imol][6] += unwrap[0];
        com[imol][7] += unwrap[1];
        com[imol][8] += unwrap[2];
      }else if(type[i] == tip_type[2]){
        com[imol][9] += unwrap[0];
        com[imol][10] += unwrap[1];
        com[imol][11] += unwrap[2];
      }
  }
  
  MPI_Allreduce(&com[0][0],&comall[0][0],12*nmolecules,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nmolecules; i++) {
    mass_inv = 1.0 / masstotal[i];
    comall[i][0] *= mass_inv;
    comall[i][1] *= mass_inv;
    comall[i][2] *= mass_inv;

    dx = comall[i][3] - comall[i][0];
    dy = comall[i][4] - comall[i][1];
    dz = comall[i][5] - comall[i][2];
    rinv = 1.0/sqrt(dx*dx + dy*dy + dz*dz);
    comall[i][3] = dx*rinv;
    comall[i][4] = dy*rinv;
    comall[i][5] = dz*rinv;

    dx = comall[i][6] - comall[i][0];
    dy = comall[i][7] - comall[i][1];
    dz = comall[i][8] - comall[i][2];
    rinv = 1.0/sqrt(dx*dx + dy*dy + dz*dz);
    comall[i][6] = dx*rinv;
    comall[i][7] = dy*rinv;
    comall[i][8] = dz*rinv;

    dx = comall[i][9] - comall[i][0];
    dy = comall[i][10] - comall[i][1];
    dz = comall[i][11] - comall[i][2];
    rinv = 1.0/sqrt(dx*dx + dy*dy + dz*dz);
    comall[i][9] = dx*rinv;
    comall[i][10] = dy*rinv;
    comall[i][11] = dz*rinv;
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

int FixDisplace::compute_molecules_in_group(int &idlo, int &idhi)
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
      if (molecule[i] == 0) error->all(FLERR,"Atom with molecule ID = 0 included in fix_displace/molecule/per::compute_molecules_in_group");
      lo = MIN(lo,molecule[i]);
      hi = MAX(hi,molecule[i]);
    }

  MPI_Allreduce(&lo,&idlo,1,MPI_INT,MPI_MIN,world);
  MPI_Allreduce(&hi,&idhi,1,MPI_INT,MPI_MAX,world);

  // molmap = vector of length nlen
  // set to 1 for IDs that appear in group across all procs, else 0

  int nlen = idhi-idlo+1;
  memory->create(molmap,nlen,"fix_displace:molmap");
  for (i = 0; i < nlen; i++) molmap[i] = 0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      molmap[molecule[i]-idlo] = 1;

  int *molmapall;
  memory->create(molmapall,nlen,"fix_displace:molmapall");
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
