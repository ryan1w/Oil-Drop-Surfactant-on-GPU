/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(chunk/atom,ComputeChunkAtom)

#else

#ifndef LMP_COMPUTE_CHUNK_ATOM_H
#define LMP_COMPUTE_CHUNK_ATOM_H

#include "compute.h"
#include <map>

namespace LAMMPS_NS {

class ComputeChunkAtom : public Compute {
 public:
  int nchunk,ncoord,compress,idsflag,lockcount;
  int computeflag;    // 1 if this compute invokes other computes
  double chunk_volume_scalar;
  double *chunk_volume_vec;
  double **coord;
  int *ichunk,*chunkID;

  ComputeChunkAtom(class LAMMPS *, int, char **);
  ~ComputeChunkAtom();
  void init();
  void setup();
  void compute_peratom();
  void set_arrays(int);
  double memory_usage();

  void lock(class Fix *, bigint, bigint);
  void unlock(class Fix *);
  int setup_chunks();
  void compute_ichunk();

 private:
  // --- NEW COMOVING TRACKERS ---
  int com_track_flag[3];
  char *com_compute_id[3];
  double com_offset[3];
  double com_shift[3];  // Tracks the Galilean atom shift
  class Compute *com_compute[3]; 
  // -----------------------------

  int which,binflag;
  int regionflag,nchunksetflag,nchunkflag,discard;
  int limit,limitstyle,limitfirst;
  int scaleflag,pbcflag;
  double xscale,yscale,zscale;
  int argindex;
  char *cfvid;

  // xyz spatial bins
  int ndim;
  int dim[3],originflag[3],nlayers[3];
  int minflag[3],maxflag[3];
  double origin[3],delta[3];
  double offset[3],invdelta[3];
  double minvalue[3],maxvalue[3];

  // spherical spatial bins
  double sorigin_user[3];
  double sorigin[3];
  double sradmin_user,sradmax_user;
  double sradmin,sradmax,sinvrad;
  int nsbin;

  // cylindrical spatial bins
  double corigin_user[3];
  double corigin[3];
  double cradmin_user,cradmax_user;
  double cradmin,cradmax,cinvrad;
  int cdim1,cdim2;
  int ncbin,ncplane;

  char *idregion;
  class Region *region;

  class Compute *cchunk;
  class Fix *fchunk;
  int vchunk;
  int maxvar;
  double *varatom;

  char *id_fix;
  class FixStore *fixstore;

  class Fix *lockfix;         
  bigint lockstart,lockstop;  

  bigint invoked_setup;    
  bigint invoked_ichunk;   
  int nmax,nmaxint;
  double *chunk;

  int molcheck;              
  int *exclude;              
  std::map<tagint,int> *hash;   

  // callback function for ring communication
  static void idring(int, char *, void *);

  void assign_chunk_ids();
  void compress_chunk_ids();
  void check_molecules();
  int setup_xyz_bins();
  int setup_sphere_bins();
  int setup_cylinder_bins();
  void bin_volumes();
  void atom2bin1d();
  void atom2bin2d();
  void atom2bin3d();
  void atom2binsphere();
  void atom2bincylinder();
  int readdim(int, char **, int, int);
};

}

#endif
#endif