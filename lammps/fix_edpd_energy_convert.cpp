/* ----------------------------------------------------------------------
     LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
     http://lammps.sandia.gov, Sandia National Laboratories
     Steve Plimpton, sjplimp@sandia.gov

     Copyright (2003) Sandia Corporation.   Under the terms of Contract
     DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
     certain rights in this software.   This software is distributed under
     the GNU General Public License.

     See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#include "math.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "group.h"
#include "error.h"
#include "fix_edpd_energy_convert.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10

FixEDPDEnergyConvert::FixEDPDEnergyConvert( LAMMPS *lmp, int narg, char **arg ) :
  Fix( lmp, narg, arg )
{
    if (narg < 4) error->all(FLERR,"Illegal fix ecvt/edpd command");

    if ( ( jgroup = group->find( arg[3] ) ) == -1 ) {
        error->all( FLERR, "fix ecvt/edpd command group ID does not exist" );
    }
    jgroupbit = group->bitmask[ jgroup ];

    theta_one = force->numeric(FLERR,arg[4]);
    da_one    = force->numeric(FLERR,arg[5]);
    omega_one = force->numeric(FLERR,arg[6]);
}

void FixEDPDEnergyConvert::init()
{
  pair = force->pair;
  cutsq = force->pair->cutsq;
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->occasional = 1;
}

void FixEDPDEnergyConvert::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

int FixEDPDEnergyConvert::setmask()
{
    int mask = 0;
    mask |= FixConst::INITIAL_INTEGRATE;
    return mask;
}

void FixEDPDEnergyConvert::initial_integrate()
{
  double **x = atom->x;
  double *T = atom->edpd_temp;
  double *Q = atom->edpd_flux;
  double *cv = atom->edpd_cv;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
 
 // invoke half neighbor list (will copy or build if necessary)
  neighbor->build_one(list);

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  double dt = update->dt;

  // loop over neighbors of my atoms
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    // skip if atom I is not in either group
    if (!(mask[i] & groupbit || mask[i] & jgroupbit)) continue;
    
    double Ti_new = T[i] + Q[i] * dt / cv[i];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int itype = type[i];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      // skip if atom J is not in either group
      if (!(mask[j] & groupbit || mask[j] & jgroupbit)) continue;

      // skip if atoms I,J are only in the same group
      int ij_flag = 0;
      int ji_flag = 0;
      if (mask[i] & groupbit && mask[j] & jgroupbit) ij_flag = 1;
      if (mask[j] & groupbit && mask[i] & jgroupbit) ji_flag = 1;
      if (!ij_flag && !ji_flag) continue;

      double delx = xtmp - x[j][0];
      double dely = ytmp - x[j][1];
      double delz = ztmp - x[j][2];
      double rsq = delx*delx + dely*dely + delz*delz;
      int jtype = type[j];

      if (rsq < cutsq[itype][jtype]){
        double r = sqrt(rsq);
        if (r < EPSILON) continue;

        double T_ij = 0.5 * (T[i] + T[j]);
        //double d_alpha = 30.0 / ( 1.0 + exp( -300.0 * ( T_ij - 1.0 ) ) ); // temperature-dependence
        double d_alpha = da_one / ( 1.0 + exp( omega_one * ( T_ij - theta_one ) ) );// temperature-dependence
        double Tj_new  = T[j] + Q[j] * dt / cv[j];
        double T_ij_new = 0.5 * ( Ti_new + Tj_new );
        //double d_alpha_new = 30.0 / ( 1.0 + exp( -300.0 * ( T_ij_new - 1.0 ) ) ); // temperature-dependence
        double d_alpha_new = da_one / ( 1.0 + exp( omega_one * ( T_ij_new - theta_one ) ) );// temperature-dependence
        double dd_alpha = d_alpha_new - d_alpha;
        double wc = 1.0 - r;
        double dq = 0.5 * dd_alpha * wc * wc / dt;
        Q[i] -= 0.5 * dq;
        if (newton_pair || j < nlocal) Q[j] -= 0.5 * dq;
      }
    }
  }
} 

int FixEDPDEnergyConvert::pack_comm( int n, int *list, double *buf, int pbc_flag, int *pbc )
{
    int m = 0;
    for( int i = 0; i < n; i++ ) {
        buf[m++] = atom->edpd_flux[ list[i] ];
    }
    return m;
}

void FixEDPDEnergyConvert::unpack_comm( int n, int first, double *buf )
{
    int m = 0;
    int last = first + n;
    for( int i = first; i < last; i++ ) {
        atom->edpd_flux[i] = buf[m++];
    }
}
