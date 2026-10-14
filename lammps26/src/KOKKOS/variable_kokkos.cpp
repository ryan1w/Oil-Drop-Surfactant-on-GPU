/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "variable_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "compute.h"
#include "fix.h"
#include "modify.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   make the per-atom arrays named by mask current on the host
------------------------------------------------------------------------- */

void VariableKokkos::sync_host(uint64_t mask)
{
  // Input is created before Atom is replaced by AtomKokkos, so the cast
  // cannot be done once in the constructor

  auto *atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (atomKK) atomKK->sync(Host, mask);
}

/* ----------------------------------------------------------------------
   compute_atom() reads atom->mask on the host for the group test
------------------------------------------------------------------------- */

void VariableKokkos::compute_atom(int ivar, int igroup, double *result, int stride, int sumflag)
{
  sync_host(MASK_MASK);
  Variable::compute_atom(ivar, igroup, result, stride, sumflag);
}

/* ----------------------------------------------------------------------
   an atom vector in a formula reads exactly one per-atom array.
   keep this list in step with Variable::atom_vector()
------------------------------------------------------------------------- */

void VariableKokkos::atom_vector(char *word, Tree **tree, Tree **treestack, int &ntreestack)
{
  uint64_t mask = ALL_MASK;

  if (strcmp(word,"id") == 0) mask = TAG_MASK;
  else if (strcmp(word,"type") == 0) mask = TYPE_MASK;
  else if (strcmp(word,"mol") == 0) mask = MOLECULE_MASK;
  else if (strcmp(word,"radius") == 0) mask = RADIUS_MASK;
  else if (strcmp(word,"q") == 0) mask = Q_MASK;
  else if (strcmp(word,"x") == 0) mask = X_MASK;
  else if (strcmp(word,"y") == 0) mask = X_MASK;
  else if (strcmp(word,"z") == 0) mask = X_MASK;
  else if (strcmp(word,"vx") == 0) mask = V_MASK;
  else if (strcmp(word,"vy") == 0) mask = V_MASK;
  else if (strcmp(word,"vz") == 0) mask = V_MASK;
  else if (strcmp(word,"fx") == 0) mask = F_MASK;
  else if (strcmp(word,"fy") == 0) mask = F_MASK;
  else if (strcmp(word,"fz") == 0) mask = F_MASK;

  // without per-atom rmass, "mass" is a per-type array that eval_tree()
  // indexes with atom->type[i], so it is atom->type that must be current

  else if (strcmp(word,"mass") == 0) mask = atom->rmass ? RMASS_MASK : TYPE_MASK;

  // an atom vector added to the base class but not listed here still works,
  // it just syncs more than it needs to

  sync_host(mask);
  Variable::atom_vector(word, tree, treestack, ntreestack);
}

/* ----------------------------------------------------------------------
   group functions such as xcm(), fcm(), and gyration() read a range of
   per-atom arrays that cannot be narrowed down from the formula
------------------------------------------------------------------------- */

int VariableKokkos::group_function(char *word, char *contents, Tree **tree, Tree **treestack,
                                   int &ntreestack, double *argstack, int &nargstack, int ivar)
{
  // group_function() is tried for every function word in a formula and returns
  // 0 for anything that is not one, so only sync when it really is one

  if (is_group_function(word)) sync_host(ALL_MASK);
  return Variable::group_function(word, contents, tree, treestack, ntreestack, argstack, nargstack,
                                  ivar);
}

/* ----------------------------------------------------------------------
   gmask() tests atom->mask, rmask() and grmask() also match a region
   against atom->x.  sum(), min(), ave(), sort() and the like take a
   compute or fix as their argument and invoke it, which reads per-atom
   data that cannot be narrowed down.  the rest touch no per-atom data.
------------------------------------------------------------------------- */

int VariableKokkos::special_function(const std::string &word, char *contents, Tree **tree,
                                     Tree **treestack, int &ntreestack, double *argstack,
                                     int &nargstack, int ivar, char *str, int &i, char *&ptr)
{
  // like group_function(), this is tried for every function word in a formula
  // and returns 0 for anything that is not a special function

  if (is_special_function(word)) {
    if (word == "gmask") sync_host(MASK_MASK);
    else if (word == "rmask") sync_host(X_MASK);
    else if (word == "grmask") sync_host(X_MASK | MASK_MASK);

    // these read nothing per-atom; anything else, including a special
    // function added later, falls through to ALL_MASK

    else if ((word != "next") && (word != "is_file") && (word != "is_os") &&
             (word != "is_timeout") && (word != "extract_setting") &&
             (word != "label2type") && (word != "is_typelabel"))
      sync_host(ALL_MASK);
  }

  return Variable::special_function(word, contents, tree, treestack, ntreestack, argstack,
                                    nargstack, ivar, str, i, ptr);
}

/* ----------------------------------------------------------------------
   access to a single atom by ID, such as x[100] or c_ID[7], reads the atom
   map as well as whichever array was named
------------------------------------------------------------------------- */

void VariableKokkos::peratom2global(int flag, char *word, double *vector, int nstride, tagint id,
                                    Tree **tree, Tree **treestack, int &ntreestack,
                                    double *argstack, int &nargstack)
{
  sync_host(ALL_MASK);
  Variable::peratom2global(flag, word, vector, nstride, id, tree, treestack, ntreestack, argstack,
                           nargstack);
}

/* ---------------------------------------------------------------------- */

void VariableKokkos::custom2global(int *ivector, double *dvector, int nstride, tagint id,
                                   Tree **tree, Tree **treestack, int &ntreestack,
                                   double *argstack, int &nargstack)
{
  sync_host(ALL_MASK);
  Variable::custom2global(ivector, dvector, nstride, id, tree, treestack, ntreestack, argstack,
                          nargstack);
}

/* ----------------------------------------------------------------------
   called from evaluate() for per-atom data reached through a compute or a
   fix, which pass the c_ID / C_ID or f_ID / F_ID word, through a custom
   property, which passes the i_ / d_ / i2_ / d2_ name the formula used, or
   through a thermo keyword, which passes a null pointer.
   a KOKKOS compute that runs on the device syncs the per-atom data it reads
   to the device itself and returns its global results on the host (and
   syncs its own per-atom results to the host), so it needs no host copy of
   the atom data; without this, every reference such as $(c_com[1]) in a
   fix print or a variable copied all modified atom arrays to the host.
   temperature computes are the exception: with a dynamic group or
   compute_modify dynamic/dof they recount their degrees of freedom on the
   host, from atom->mask and the dof() of every fix, so they get everything.
   a KOKKOS fix that runs on the device and has no per-atom output keeps its
   global output on the host (summed in its own hooks, or computed from data
   it syncs itself, as fix langevin/kk does), so a reference such as
   $(f_ID[2]) needs no host copy either.
   per-atom access by atom ID goes through peratom2global(), which syncs.
   any other compute or fix, including a KOKKOS one running on the host,
   can read host data directly and gets everything.
------------------------------------------------------------------------- */

void VariableKokkos::sync_peratom(const char *word)
{
  uint64_t mask = ALL_MASK;

  if (!word) mask = ALL_MASK;
  else if (((word[0] == 'c') || (word[0] == 'C')) && (word[1] == '_')) {
    Compute *compute = modify->get_compute_by_id(word+2);
    if (compute && compute->kokkosable && (compute->execution_space == Device) &&
        !compute->tempflag)
      return;
    mask = ALL_MASK;
  } else if (((word[0] == 'f') || (word[0] == 'F')) && (word[1] == '_')) {
    Fix *fix = modify->get_fix_by_id(word+2);
    if (fix && fix->kokkosable && (fix->execution_space == Device) && !fix->peratom_flag) return;
    mask = ALL_MASK;
  } else if (strncmp(word,"i2_",3) == 0) mask = IARRAY_MASK;
  else if (strncmp(word,"d2_",3) == 0) mask = DARRAY_MASK;
  else if (strncmp(word,"i_",2) == 0) mask = IVECTOR_MASK;
  else if (strncmp(word,"d_",2) == 0) mask = DVECTOR_MASK;

  sync_host(mask);
}
