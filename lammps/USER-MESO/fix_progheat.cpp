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

/* Exerting force to induce the periodic Poiseuille flow
 * Adapted from the CPU version fix_zl_force first written by:
 * Zhen Li, Crunch Group, Division of Applied Mathematics, Brown University
 * June, 2013
 */
#include "stdio.h"
#include "string.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "input.h"
#include "variable.h"
#include "fix_progheat.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixProgHeat::FixProgHeat( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    t_user("@t"),
    T_varname("__PNIPAM_internal_programmable_temperature__")
{
    if( narg < 5 ) error->all( FLERR, "Illegal fix progheat command" );

    scalar_flag = 1;
    extscalar = 0;
    T_command = new char*[3];
    for(int i = 0 ; i < 3 ; i++) T_command[i] = new char[128];
    strcpy( T_command[0], T_varname.c_str() );
    strcpy( T_command[1], "equal" );

    delta = 0.;
    Cv = 0.;

    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "T" ) ) {
            T_template = arg[++i];
            continue;
        }
        if( !strcmp( arg[i], "delta" ) ) {
            delta = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "Cv" ) ) {
            Cv = atof( arg[++i] );
            continue;
        }
    }

    if ( T_template == "" ) error->all( FLERR, "fix progheat usage: [Cv double] [T expression] [delta double]" );
    if ( Cv == 0. ) error->warning( FLERR, "fix progheat Cv = 0, cannot calculate heat flux" );
}

FixProgHeat::~FixProgHeat() {
    for(int i = 0; i < 3; i++) delete [] T_command[i];
    delete [] T_command;
}

int FixProgHeat::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void FixProgHeat::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "RESPA not supported in Fix_ProgHeat. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

void FixProgHeat::post_force( int vflag )
{
    // calculate current time
    double t = double( update->ntimestep ) * update->dt;
    // substitute time into temperature function
    char t_current[256];
    sprintf( t_current, "%le", t );
    std::string t_str( t_current );
    std::string T_expr( T_template );
    while( T_expr.find(t_user) != std::string::npos ) {
        T_expr.replace( T_expr.find(t_user), t_user.size(), t_str );
    }
    // update LAMMPS variable for broadcasting
    strcpy( T_command[2], T_expr.c_str() );
    input->variable->set( 3, T_command );
    // evaluate
    double T0 = input->variable->compute_equal( input->variable->find( (char*)T_varname.c_str() ) );
    /*if ( update->ntimestep%100==0) {
        fprintf( stderr, "%s = %12.15lf\n", T_expr.c_str(), T0 );
        gpu_reduce_sum_host<double><<<1,1024,0,meso_device->stream()>>>( meso_atom->dev_T, meanT.ptr(), atom->nlocal );
        meso_device->sync_device();
        printf("mean T = %.4lf\n", meanT.ptr()[0] / atom->nlocal );
    }*/
    dQ = 0.0;
    double *edpd_temp = atom->edpd_temp;
    double *edpd_flux = atom->edpd_flux;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;
    
    for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
        double dT = ( T0 - edpd_temp[i] ) * delta;
        edpd_temp[i] += dT;
        edpd_flux[i] += Cv * dT;
        dQ += Cv * dT;
    }
}

double FixProgHeat::compute_scalar()
{
    double scalar;
     MPI_Allreduce( &dQ, &scalar, 1, MPI_DOUBLE, MPI_SUM, world );
    return scalar;
}
