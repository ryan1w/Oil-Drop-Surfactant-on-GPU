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
#include <cstring>
#include <cstdlib>
#include "atom_vec_edpd_bond.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "update.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecEDPDBond::AtomVecEDPDBond(LAMMPS *lmp) : AtomVec(lmp)
{
  if(strcmp(update->unit_style,"lj") != 0)
    error->all(FLERR,"Atom style edpd requires lj units");

  molecular = 1;
  bonds_allow = 1;
  mass_type = 1;
  forceclearflag = 1;

  comm_x_only = comm_f_only = 0;
  comm->ghost_velocity = 1;

  size_forward = 3 + 9; // edpd_temp + vest[7] + phi
  size_reverse = 3 + 1;     // edpd_flux
  size_border = 7 + 10;  // edpd_temp + edpd_cv + vest[7] + phi
  size_velocity = 3;
  size_data_atom = 5 + 3; // we read atom-id + molecule-id + atom-type + xyz[3] + edpd_temp + edpd_cv
  size_data_vel = 4;
  xcol_data = 4;

  atom->edpd_flag = 1;
  atom->vest_flag = 1;
  atom->molecule_flag = 1;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecEDPDBond::grow(int n)
{
  if (n == 0) grow_nmax();
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  tag = memory->grow(atom->tag,nmax,"atom:tag");
  type = memory->grow(atom->type,nmax,"atom:type");
  mask = memory->grow(atom->mask,nmax,"atom:mask");
  image = memory->grow(atom->image,nmax,"atom:image");
  x = memory->grow(atom->x,nmax,3,"atom:x");
  v = memory->grow(atom->v,nmax,3,"atom:v");
  f = memory->grow(atom->f,nmax*comm->nthreads,3,"atom:f");
  edpd_cv= memory->grow(atom->edpd_cv, nmax, "atom:edpd_cv");
  edpd_temp = memory->grow(atom->edpd_temp, nmax, "atom:edpd_temp");
  edpd_flux = memory->grow(atom->edpd_flux, nmax*comm->nthreads,"atom:edpd_flux");
  vest = memory->grow(atom->vest, nmax, 7, "atom:vest");

  phi = memory->grow(atom->phi, nmax*comm->nthreads, "atom:phi");
  nw = memory->grow(atom->nw, nmax*comm->nthreads, 3, "atom:nw");

  molecule = memory->grow(atom->molecule,nmax,"atom:molecule");
  nspecial = memory->grow(atom->nspecial,nmax,3,"atom:nspecial");
  special = memory->grow(atom->special,nmax,atom->maxspecial,"atom:special");
  num_bond = memory->grow(atom->num_bond,nmax,"atom:num_bond");
  bond_type = memory->grow(atom->bond_type,nmax,atom->bond_per_atom,"atom:bond_type");
  bond_atom = memory->grow(atom->bond_atom,nmax,atom->bond_per_atom,"atom:bond_atom");

  if (atom->nextra_grow)
  for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
    modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecEDPDBond::grow_reset()
{
  tag = atom->tag; type = atom->type;
  mask = atom->mask; image = atom->image;
  x = atom->x; v = atom->v; f = atom->f;
  edpd_cv = atom->edpd_cv; edpd_temp = atom->edpd_temp; edpd_flux = atom->edpd_flux; vest = atom->vest;
  molecule = atom->molecule;
  nspecial = atom->nspecial; special = atom->special;
  num_bond = atom->num_bond; bond_type = atom->bond_type;
  bond_atom = atom->bond_atom;
  phi = atom->phi;
  nw = atom->nw;
  vest = atom->vest;

}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
------------------------------------------------------------------------- */

void AtomVecEDPDBond::copy(int i, int j, int delflag)
{
  tag[j] = tag[i];
  type[j] = type[i];
  mask[j] = mask[i];
  image[j] = image[i];
  x[j][0] = x[i][0];
  x[j][1] = x[i][1];
  x[j][2] = x[i][2];
  v[j][0] = v[i][0];
  v[j][1] = v[i][1];
  v[j][2] = v[i][2];

  edpd_temp[j] = edpd_temp[i];
  edpd_flux[j] = edpd_flux[i];
  edpd_cv[j] = edpd_cv[i];
  vest[j][0] = vest[i][0];
  vest[j][1] = vest[i][1];
  vest[j][2] = vest[i][2];
  vest[j][3] = vest[i][3];

  phi[j] = phi[i];
  for(int k = 0; k < 3; k++){
    nw[j][k] = nw[i][k];
    vest[j][k+4] = vest[i][k+4];
  }


  molecule[j] = molecule[i];
  num_bond[j] = num_bond[i];
  for(int k = 0; k < num_bond[j]; k++) {
      bond_type[j][k] = bond_type[i][k];
      bond_atom[j][k] = bond_atom[i][k];
  }
  
  nspecial[j][0] = nspecial[i][0];
  nspecial[j][1] = nspecial[i][1];
  nspecial[j][2] = nspecial[i][2];
  for(int k = 0; k < nspecial[j][2]; k++) special[j][k] = special[i][k];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i,j,delflag);
}


void AtomVecEDPDBond::force_clear(int n, size_t nbytes)
{
  memset(&edpd_flux[n],0,nbytes);
}


/* ---------------------------------------------------------------------- */

int AtomVecEDPDBond::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = edpd_temp[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      buf[m++] = vest[j][3];
      
      buf[m++] = vest[j][4];
      buf[m++] = vest[j][5];
      buf[m++] = vest[j][6];	
      buf[m++] = phi[j];

    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = edpd_temp[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      buf[m++] = vest[j][3];
      
      buf[m++] = vest[j][4];
      buf[m++] = vest[j][5];
      buf[m++] = vest[j][6];
      buf[m++] = phi[j];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecEDPDBond::pack_comm_vel(int n, int *list, double *buf,
                                 int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = edpd_temp[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      buf[m++] = vest[j][3];
      
      buf[m++] = vest[j][4];
      buf[m++] = vest[j][5];
      buf[m++] = vest[j][6];
      buf[m++] = phi[j];

    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
        buf[m++] = edpd_temp[j];
        buf[m++] = vest[j][0];
        buf[m++] = vest[j][1];
        buf[m++] = vest[j][2];
        buf[m++] = vest[j][3];
	
	buf[m++] = vest[j][4];
        buf[m++] = vest[j][5];
        buf[m++] = vest[j][6];
        buf[m++] = phi[j];
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
        }
        buf[m++] = edpd_temp[j];
        buf[m++] = vest[j][0];
        buf[m++] = vest[j][1];
        buf[m++] = vest[j][2];
        buf[m++] = vest[j][3];
	
	buf[m++] = vest[j][4];
        buf[m++] = vest[j][5];
        buf[m++] = vest[j][6];
        buf[m++] = phi[j];
      }
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecEDPDBond::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    edpd_temp[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    vest[i][3] = buf[m++];
    
    vest[i][4] = buf[m++];
    vest[i][5] = buf[m++];
    vest[i][6] = buf[m++];
    phi[i] = buf[m++];

  }
}

/* ---------------------------------------------------------------------- */

void AtomVecEDPDBond::unpack_comm_vel(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    edpd_temp[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    vest[i][3] = buf[m++];

    vest[i][4] = buf[m++];
    vest[i][5] = buf[m++];
    vest[i][6] = buf[m++];
    phi[i] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecEDPDBond::pack_reverse(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
    buf[m++] = edpd_flux[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecEDPDBond::unpack_reverse(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
    edpd_flux[j] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecEDPDBond::pack_border(int n, int *list, double *buf,
                               int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = edpd_temp[j];
      buf[m++] = edpd_cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      buf[m++] = vest[j][3];
      buf[m++] = ubuf(molecule[j]).d;

      buf[m++] = vest[j][4];
      buf[m++] = vest[j][5];
      buf[m++] = vest[j][6];
      buf[m++] = phi[j];

    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = edpd_temp[j];
      buf[m++] = edpd_cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      buf[m++] = vest[j][3];
      buf[m++] = ubuf(molecule[j]).d;

      buf[m++] = vest[j][4];
      buf[m++] = vest[j][5];
      buf[m++] = vest[j][6];
      buf[m++] = phi[j];

    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecEDPDBond::pack_border_vel(int n, int *list, double *buf,
                                   int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = edpd_temp[j];
      buf[m++] = edpd_cv[j];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      buf[m++] = vest[j][3];
      buf[m++] = ubuf(molecule[j]).d;

      buf[m++] = vest[j][4];
      buf[m++] = vest[j][5];
      buf[m++] = vest[j][6];
      buf[m++] = phi[j];

    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = edpd_temp[j];
        buf[m++] = edpd_cv[j];
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
        buf[m++] = vest[j][0];
        buf[m++] = vest[j][1];
        buf[m++] = vest[j][2];
        buf[m++] = vest[j][3];
        buf[m++] = ubuf(molecule[j]).d;

	buf[m++] = vest[j][4];
        buf[m++] = vest[j][5];
        buf[m++] = vest[j][6];
        buf[m++] = phi[j];
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = edpd_temp[j];
        buf[m++] = edpd_cv[j];
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
          buf[m++] = vest[j][0] + dvx;
          buf[m++] = vest[j][1] + dvy;
          buf[m++] = vest[j][2] + dvz;
          buf[m++] = vest[j][3];

	  buf[m++] = phi[j];
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
          buf[m++] = vest[j][0];
          buf[m++] = vest[j][1];
          buf[m++] = vest[j][2];
          buf[m++] = vest[j][3];
		
	  buf[m++] = vest[j][4];
          buf[m++] = vest[j][5];
          buf[m++] = vest[j][6];
          buf[m++] = phi[j];

        }
        buf[m++] = ubuf(molecule[j]).d;
      }
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecEDPDBond::unpack_border(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax) grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    edpd_temp[i] = buf[m++];
    edpd_cv[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    vest[i][3] = buf[m++];
    molecule[i] = (tagint) ubuf(buf[m++]).i;

    vest[i][4] = buf[m++];
    vest[i][5] = buf[m++];
    vest[i][6] = buf[m++];
    phi[i] = buf[m++];

  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

void AtomVecEDPDBond::unpack_border_vel(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax) grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    edpd_temp[i] = buf[m++];
    edpd_cv[i] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    vest[i][3] = buf[m++];
    molecule[i] = (tagint) ubuf(buf[m++]).i;
	
    vest[i][4] = buf[m++];
    vest[i][5] = buf[m++];
    vest[i][6] = buf[m++];
    phi[i] = buf[m++];

  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them
------------------------------------------------------------------------- */

int AtomVecEDPDBond::pack_exchange(int i, double *buf)
{
  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  
  buf[m++] = edpd_temp[i];
  buf[m++] = edpd_cv[i];
  buf[m++] = vest[i][0];
  buf[m++] = vest[i][1];
  buf[m++] = vest[i][2];
  buf[m++] = vest[i][3];

  buf[m++] = vest[i][4];
  buf[m++] = vest[i][5];
  buf[m++] = vest[i][6];
  buf[m++] = phi[i];


  buf[m++] = ubuf(molecule[i]).d;
  buf[m++] = ubuf(num_bond[i]).d;
  for (int k = 0; k < num_bond[i]; k++) {
    buf[m++] = ubuf(bond_type[i][k]).d;
    buf[m++] = ubuf(bond_atom[i][k]).d;
  }

  buf[m++] = ubuf(nspecial[i][0]).d;
  buf[m++] = ubuf(nspecial[i][1]).d;
  buf[m++] = ubuf(nspecial[i][2]).d;
  for (int k = 0; k < nspecial[i][2]; k++) buf[m++] = ubuf(special[i][k]).d;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i,&buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecEDPDBond::unpack_exchange(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;

  edpd_temp[nlocal] = buf[m++];
  edpd_cv[nlocal] = buf[m++];
  vest[nlocal][0] = buf[m++];
  vest[nlocal][1] = buf[m++];
  vest[nlocal][2] = buf[m++];
  vest[nlocal][3] = buf[m++];

  vest[nlocal][4] = buf[m++];
  vest[nlocal][5] = buf[m++];
  vest[nlocal][6] = buf[m++];
  phi[nlocal] = buf[m++];

  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;
  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
  for (int k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (tagint) ubuf(buf[m++]).i;
  }

  nspecial[nlocal][0] = (int) ubuf(buf[m++]).i;
  nspecial[nlocal][1] = (int) ubuf(buf[m++]).i;
  nspecial[nlocal][2] = (int) ubuf(buf[m++]).i;
  for (int k = 0; k < nspecial[nlocal][2]; k++) special[nlocal][k] = (tagint) ubuf(buf[m++]).i;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->
        unpack_exchange(nlocal,&buf[m]);

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
------------------------------------------------------------------------- */

int AtomVecEDPDBond::size_restart()
{
  int n = 0;
    for( int i = 0; i < atom->nlocal; i++ ) n += 11 + 12 + 2 * num_bond[i];
      //11 + edpd_temp + edpd_cv + vest[7] + molecule + num_bond + phi
  if (atom->nextra_restart)
    for(int iextra = 0; iextra < atom->nextra_restart; iextra++)
      for(int i = 0; i < atom->nlocal; i++)
        n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

  return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
------------------------------------------------------------------------- */

int AtomVecEDPDBond::pack_restart(int i, double *buf)
{
  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = edpd_temp[i];
  buf[m++] = edpd_cv[i];
  buf[m++] = vest[i][0];
  buf[m++] = vest[i][1];
  buf[m++] = vest[i][2];
  buf[m++] = vest[i][3];

  buf[m++] = vest[i][4];
  buf[m++] = vest[i][5];
  buf[m++] = vest[i][6];
  buf[m++] = phi[i];

  buf[m++] = ubuf(molecule[i]).d;

  buf[m++] = ubuf(num_bond[i]).d;
  for (int k = 0; k < num_bond[i]; k++) {
    buf[m++] = ubuf(MAX(bond_type[i][k],-bond_type[i][k])).d;
    buf[m++] = ubuf(bond_atom[i][k]).d;
  }

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i,&buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
------------------------------------------------------------------------- */

int AtomVecEDPDBond::unpack_restart(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store)
      memory->grow(atom->extra,nmax,atom->nextra_store,"atom:extra");
  }

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;

  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];

  edpd_temp[nlocal] = buf[m++];
  edpd_cv[nlocal]= buf[m++];
  vest[nlocal][0] = buf[m++];
  vest[nlocal][1] = buf[m++];
  vest[nlocal][2] = buf[m++];
  vest[nlocal][3] = buf[m++];

  vest[nlocal][4] = buf[m++];
  vest[nlocal][5] = buf[m++];
  vest[nlocal][6] = buf[m++];
  phi[nlocal] = buf[m++];


  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;

  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
  for (int k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (tagint) ubuf(buf[m++]).i;
  }

  nspecial[nlocal][0] = nspecial[nlocal][1] = nspecial[nlocal][2] = 0;

  double **extra = atom->extra;
  if (atom->nextra_store) {
    int size = static_cast<int> (buf[0]) - m;
    for (int i = 0; i < size; i++) extra[nlocal][i] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
------------------------------------------------------------------------- */

void AtomVecEDPDBond::create_atom(int itype, double *coord)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];
  mask[nlocal] = 1;
  image[nlocal] = ((imageint) IMGMAX << IMG2BITS) |
    ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  edpd_temp[nlocal] = 1.0;
  edpd_flux[nlocal] = 0.0;
  edpd_cv[nlocal]= 1.0E3;
  vest[nlocal][0] = 0.0;
  vest[nlocal][1] = 0.0;
  vest[nlocal][2] = 0.0;
  vest[nlocal][3] = edpd_temp[nlocal];

  vest[nlocal][4] = 0.0;
  vest[nlocal][5] = 0.0;
  vest[nlocal][6] = 0.0;
  phi[nlocal] = 0.0;
  nw[nlocal][0] = 0.0;
  nw[nlocal][1] = 0.0;
  nw[nlocal][2] = 0.0;

  molecule[nlocal] = 0;
  num_bond[nlocal] = 0;
  nspecial[nlocal][0] = nspecial[nlocal][1] = nspecial[nlocal][2] = 0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecEDPDBond::data_atom(double *coord, imageint imagetmp, char **values)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = ATOTAGINT(values[0]);
  if( tag[nlocal] <= 0 )
        error->one( FLERR, "Invalid atom ID in Atoms section of data file" );
        
  molecule[nlocal] = atoi(values[1]);

  type[nlocal] = atoi(values[2]);
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR,"Invalid atom type in Atoms section of data file");

  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];

  edpd_temp[nlocal] = atof(values[6]);

  edpd_cv[nlocal] = atof(values[7]);
  image[nlocal] = imagetmp;

  mask[nlocal] = 1;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  vest[nlocal][0] = 0.0;
  vest[nlocal][1] = 0.0;
  vest[nlocal][2] = 0.0;
  vest[nlocal][3] = edpd_temp[nlocal];

  vest[nlocal][4] = 0.0;
  vest[nlocal][5] = 0.0;
  vest[nlocal][6] = 0.0;
  phi[nlocal] = 0.0;
  nw[nlocal][0] = 0.0;
  nw[nlocal][1] = 0.0;
  nw[nlocal][2] = 0.0;

  edpd_flux[nlocal] = 0.0;
  num_bond[nlocal] = 0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecEDPDBond::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]).d;
    buf[i][1] = ubuf(molecule[i]).d;
    buf[i][2] = ubuf(type[i]).d;
    buf[i][3] = x[i][0];
    buf[i][4] = x[i][1];
    buf[i][5] = x[i][2];
    buf[i][6] = edpd_temp[i];
    buf[i][7] = edpd_cv[i];
    buf[i][8] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][9] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][10] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;
  }
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecEDPDBond::write_data(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp,TAGINT_FORMAT " %d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d\n",
            (tagint) ubuf(buf[i][0]).i,(int) ubuf(buf[i][1]).i,(int) ubuf(buf[i][2]).i,
            buf[i][3],buf[i][4],buf[i][5],buf[i][6],buf[i][7],
            (int) ubuf(buf[i][8]).i,(int) ubuf(buf[i][9]).i,(int) ubuf(buf[i][10]).i);
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

bigint AtomVecEDPDBond::memory_usage()
{
  bigint bytes = 0;

  if (atom->memcheck("tag")) bytes += memory->usage(tag,nmax);
  if (atom->memcheck("type")) bytes += memory->usage(type,nmax);
  if (atom->memcheck("mask")) bytes += memory->usage(mask,nmax);
  if (atom->memcheck("image")) bytes += memory->usage(image,nmax);
  if (atom->memcheck("x")) bytes += memory->usage(x,nmax,3);
  if (atom->memcheck("v")) bytes += memory->usage(v,nmax,3);
  if (atom->memcheck("f")) bytes += memory->usage(f,nmax*comm->nthreads,3);

  if (atom->memcheck("molecule")) bytes += memory->usage(molecule,nmax);
  if (atom->memcheck("nspecial")) bytes += memory->usage(nspecial,nmax,3);
  if (atom->memcheck("special"))
    bytes += memory->usage(special,nmax,atom->maxspecial);

  if (atom->memcheck("num_bond")) bytes += memory->usage(num_bond,nmax);
  if (atom->memcheck("bond_type"))
    bytes += memory->usage(bond_type,nmax,atom->bond_per_atom);
  if (atom->memcheck("bond_atom"))
    bytes += memory->usage(bond_atom,nmax,atom->bond_per_atom);
  if (atom->memcheck("edpd_temp")) bytes += memory->usage(edpd_temp,nmax);
  if (atom->memcheck("edpd_flux")) bytes += memory->usage(edpd_flux,nmax*comm->nthreads);
  if (atom->memcheck("edpd_cv")) bytes += memory->usage(edpd_cv,nmax);
  if (atom->memcheck("vest")) bytes += memory->usage(vest,nmax,7);

  if (atom->memcheck("phi")) bytes += memory->usage(phi, nmax*comm->nthreads);
  if (atom->memcheck("nw")) bytes += memory->usage(nw, nmax*comm->nthreads, 3);


  return bytes;
}
