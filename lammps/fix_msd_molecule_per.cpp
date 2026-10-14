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
   This fix is designed for computing the msd of molecules
------------------------------------------------------------------------- */
#include "atom.h"
#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "fix_msd_molecule_per.h"
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

FixMsdMoleculePer::FixMsdMoleculePer(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix_msd/molecule/per command");

  Nstart = force->inumeric(FLERR,arg[3]);
  Nfreq = force->inumeric(FLERR,arg[4]);
  Nlength = force->inumeric(FLERR,arg[5]);

  if (atom->molecular == 0)
    error->all(FLERR,"fix_msd/molecule/per requires molecular atom style");
 
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  molmap = NULL;
  nmolecules = compute_molecules_in_group(idlo,idhi);
//  cout << nmolecules <<"number of mol\n";

  memory->create(massproc,nmolecules,"fix_msd/molecule/per:massproc");
  memory->create(masstotal,nmolecules,"fix_msd/molecule/per:masstotal");
  memory->create(com,nmolecules,3,"fix_msd/molecule/per:com");
  memory->create(comall,nmolecules,3,"fix_msd/molecule/per:comall");
//  memory->create(cominit,size,nmolecules,3,"fix_msd/molecule:cominit");

  size = int(Nlength/Nfreq);
  memory->create(cominit,size,nmolecules,3,"fix_msd/molecule/per:cominit");
  memory->create(msd,nmolecules,size,2,"fix_msd/molecule/per:msd");
  memory->create(msd_bin,size,1,"fix_msd/molecule/per:msd");
  
  for(int k = 0; k < nmolecules; k++)
    for(int i = 0; i < size; i++) {
      msd[k][i][0] = 0.0; msd[k][i][1] = 0.0;
    }

  for (int i = 0; i < size; i++)
	  msd_bin[i][0] = 0.0;
  
  // compute masstotal for each molecule
  int imol;
  double massone;

  for (int i = 0; i < nmolecules; i++) massproc[i] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      //imol = molecule[i]-1;
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
    for (int j = 0; j < 3; j++) 
      cominit[0][i][j] = comall[i][j];
  
  firstflag = 0;

  //=======================================================================
  if(comm->me == 0) {
    fpmsd.open("msd_mol_per.dat",ios::app | ios::out);
    fpmsd << "Variables = \"time\", \"msd\"\n";
  }
  //=======================================================================
}

/* ---------------------------------------------------------------------- */

FixMsdMoleculePer::~FixMsdMoleculePer()
{
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(cominit);
  memory->destroy(msd);
  memory->destroy(molmap);

//  if (comm->me == 0) {
//    fpmsd.close();
//  }
}

/* ---------------------------------------------------------------------- */

int FixMsdMoleculePer::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMsdMoleculePer::end_of_step()
{
  bigint ntimestep = update->ntimestep;
  int step, loop;

  if(ntimestep > Nstart) {
    step = (ntimestep - Nstart) % Nlength;
    loop = (ntimestep - Nstart) / Nlength;
    if(step % Nfreq == 0) {
      double dx, dy, dz;
      //double xprd = domain->xprd;
      //double yprd = domain->yprd;
      //double zprd = domain->zprd;
  
      compute_com();
      if (firstflag) error->all(FLERR,"fix_msd_molecule_per error. cominit has not been set yet.");

      int index = step / Nfreq;
      for (int i = 0; i < nmolecules; i++)
        for (int j = 0; j < 3; j++)
          cominit[index][i][j] = comall[i][j];
               
      if(comm->me == 0) {
        for(int i = 0; i < nmolecules; i++)
          for(int k = 0; k <= index; k++) {
            dx = comall[i][0] - cominit[k][i][0];
            dy = comall[i][1] - cominit[k][i][1];
            dz = comall[i][2] - cominit[k][i][2];
            //if(fabs(dx) >= 0.5 * xprd) dx += ((dx > 0.0) ? -1.0 : 1.0) * xprd;
            //if(fabs(dy) >= 0.5 * yprd) dy += ((dy > 0.0) ? -1.0 : 1.0) * yprd;
            //if(fabs(dz) >= 0.5 * zprd) dz += ((dz > 0.0) ? -1.0 : 1.0) * zprd;
            //minimum_image(dx,dy,dz);
            
            msd[i][index-k][0] += dx*dx + dy*dy + dz*dz;
            msd[i][index-k][1] += 1;
          }
          
        if(loop > 0)
          for(int i = 0; i < nmolecules; i++)
            for(int k = index+1; k < size; k++) {
              dx = comall[i][0] - cominit[k][i][0];
              dy = comall[i][1] - cominit[k][i][1];
              dz = comall[i][2] - cominit[k][i][2];
              //if(fabs(dx) >= 0.5 * xprd) dx += ((dx > 0.0) ? -1.0 : 1.0) * xprd;
              //if(fabs(dy) >= 0.5 * yprd) dy += ((dy > 0.0) ? -1.0 : 1.0) * yprd;
              //if(fabs(dz) >= 0.5 * zprd) dz += ((dz > 0.0) ? -1.0 : 1.0) * zprd;
              //minimum_image(dx,dy,dz);

              msd[i][size+index-k][0] += dx*dx + dy*dy + dz*dz;
              msd[i][size+index-k][1] += 1;
            }
      }
    }
  }
  
  if(ntimestep == update->laststep && comm->me == 0) {
	  for(int tau = 0; tau < size; tau++)
		  for(int k = 0; k < nmolecules; k++) 
			  msd_bin[tau][0] += (msd[k][tau][0]/msd[k][tau][1]);
	  fpmsd << "zone i ="<< size << ",f=point" << endl;
	  for(int tau = 0; tau < size; tau++)
		  fpmsd << tau*Nfreq*update->dt <<"  "<< msd_bin[tau][0] / nmolecules <<endl;
    }
}

// compute current COM positions
void FixMsdMoleculePer::compute_com()
{
  int imol;
  double massone, mass_inv;
  double unwrap[3];

  for (int i = 0; i < nmolecules; i++)
    for (int j = 0; j < 3; j++)
      com[i][j] = 0.0;

  double **x = atom->x;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int *type = atom->type;
  tagint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
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
    }
  }
	MPI_Allreduce(&com[0][0],&comall[0][0],3*nmolecules,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nmolecules; i++) {
    mass_inv = 1.0 / masstotal[i];
    comall[i][0] *= mass_inv;
    comall[i][1] *= mass_inv;
    comall[i][2] *= mass_inv;
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

int FixMsdMoleculePer::compute_molecules_in_group(int &idlo, int &idhi)
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
      if (molecule[i] == 0) error->all(FLERR,"Atom with molecule ID = 0 included in fix_msd/molecule/per::compute_molecules_in_group");
      lo = MIN(lo,molecule[i]);
      hi = MAX(hi,molecule[i]);
    }

  MPI_Allreduce(&lo,&idlo,1,MPI_INT,MPI_MIN,world);
  MPI_Allreduce(&hi,&idhi,1,MPI_INT,MPI_MAX,world);

  // molmap = vector of length nlen
  // set to 1 for IDs that appear in group across all procs, else 0

  int nlen = idhi-idlo+1;
  memory->create(molmap,nlen,"fix_msd/molecule/per:molmap");
  for (i = 0; i < nlen; i++) molmap[i] = 0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      molmap[molecule[i]-idlo] = 1;

  int *molmapall;
  memory->create(molmapall,nlen,"fix_msd/molecule/per:molmapall");
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
