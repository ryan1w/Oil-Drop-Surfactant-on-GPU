// clang-format off
/* LAMMPS spherical solvent boundary, 2026.
   Distributed under the GNU General Public License.

   This is a documented extension inspired by Li & Drazer (2008), not
   their original code. Radial position projection is a finite-step
   geometric approximation. It is not an exact hard-sphere propagator.
   The fixed-position velocity operation is a conditional Maxwell
   heatbath with finite-mass translation and rotation of one rigid body.
   Projection follows the position drift (POST_INTEGRATE); thermalization
   follows the final force kick (END_OF_STEP). Do not move the heatbath to
   POST_INTEGRATE: randomizing half-step velocities before the last kick
   biases configurational equilibrium even for a conservative harmonic well.

   Syntax:
   fix ID fluid fluid/particle/treatment rigidID R delta T seed
       [mode thermal|penetration] [maxpenetration fraction]

   Defaults: mode thermal, maxpenetration 0.25 (fraction of R).
   The sphere's geometric center is its rigid-body center of mass.
   Supports one rigid/nve/small body, plain nve solvent, units lj,
   three-dimensional orthogonal fixed boxes, all boundaries periodic.
   This fix requires the plain styles (verlet, nve, rigid/nve/small): their
   KOKKOS variants keep the atoms and the body on the device, where the host
   loops of this fix would read and write stale data.  With KOKKOS use
   fix fluid/particle/treatment/kk (src/KOKKOS), which requires the /kk
   integrators and replaces the per-atom loops (scan_setup, collect_contacts,
   apply_contacts) with device kernels; the collective contact processing
   in process_boundary() is shared, and so are the heatbath variates
   (gaussian3() below, drawn on the host for the few contacts).  The /kk
   fix and its rigid/nve/small/kk fix must run in the same execution space.
 */

#include "fix_fluid_particle_treatment.h"
#include "fluid_particle_treatment_math.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_rigid_small.h"
#include "modify.h"
#include "update.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace FluidParticleMath;

namespace {

using Contact=FixFluidParticleTreatment::Contact;

bool contact_less(const Contact &a,const Contact &b) { return a.tag<b.tag; }
bool finite_value(double x) { return x==x && std::fabs(x)<=DBL_MAX; }

// Stateless counter RNG. The keys are seed, absolute timestep, atom ID,
// and lane. No process/rank/index/iteration-order-dependent random state.
// The unsigned arithmetic intentionally wraps modulo 2^64.
uint64_t mix64(uint64_t x)
{
  x+=UINT64_C(0x9e3779b97f4a7c15);
  x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
  return x^(x>>31);
}

double uniform_open(uint64_t key,uint64_t lane)
{
  const uint64_t bits=mix64(key+UINT64_C(0x9e3779b97f4a7c15)*lane);
  return (static_cast<double>(bits>>12)+0.5)/4503599627370496.0;
}

void gaussian3(int seed,bigint timestep,tagint tag,double *z)
{
  uint64_t key=mix64(static_cast<uint64_t>(seed));
  key=mix64(key^static_cast<uint64_t>(timestep));
  key=mix64(key^static_cast<uint64_t>(tag));
  const double twopi=6.283185307179586476925286766559;
  double a=std::sqrt(-2.0*std::log(uniform_open(key,1)));
  double theta=twopi*uniform_open(key,2);
  z[0]=a*std::cos(theta);
  z[1]=a*std::sin(theta);
  a=std::sqrt(-2.0*std::log(uniform_open(key,3)));
  theta=twopi*uniform_open(key,4);
  z[2]=a*std::cos(theta);
}

double pair_ke(const Contact &c,const FixRigidSmall::FPTState &s)
{
  return 0.5*c.mass*dot(c.v,c.v)+0.5*s.mass*dot(s.vcm,s.vcm)+
         rotational_energy(s.inverse_inertia,s.angmom);
}

} // anonymous namespace

FixFluidParticleTreatment::FixFluidParticleTreatment(LAMMPS *lmp,
    int narg,char **arg) : Fix(lmp,narg,arg), rigid_id(nullptr), rigid(nullptr),
    kokkos_variant(0)
{
  if (narg<8) utils::missing_cmd_args(FLERR,"fix fluid/particle/treatment",error);

  radius=utils::numeric(FLERR,arg[4],false,lmp);
  delta=utils::numeric(FLERR,arg[5],false,lmp);
  temperature=utils::numeric(FLERR,arg[6],false,lmp);
  seed=utils::inumeric(FLERR,arg[7],false,lmp);
  thermal=1;
  maxpenetration=0.25;
  int imaxpen=Error::NOPOINTER;   // argument index of maxpenetration value
  for (int i=8;i<narg;) {
    if (!strcmp(arg[i],"mode")) {
      if (i+1>=narg) utils::missing_cmd_args(FLERR,"fix fluid/particle/treatment mode",error);
      if (!strcmp(arg[i+1],"thermal")) thermal=1;
      else if (!strcmp(arg[i+1],"penetration")) thermal=0;
      else error->all(FLERR,i+1,"Fix fluid/particle/treatment mode must be thermal or penetration");
      i+=2;
    } else if (!strcmp(arg[i],"maxpenetration")) {
      if (i+1>=narg)
        utils::missing_cmd_args(FLERR,"fix fluid/particle/treatment maxpenetration",error);
      maxpenetration=utils::numeric(FLERR,arg[i+1],false,lmp);
      imaxpen=i+1;
      i+=2;
    } else error->all(FLERR,i,"Unknown fix fluid/particle/treatment keyword: {}",arg[i]);
  }
  // Same conditions as the single combined check of the 2018 version.
  if (!finite_value(radius) || !(radius>0))
    error->all(FLERR,4,"Fix fluid/particle/treatment sphere radius R must be positive and finite");
  if (!finite_value(delta) || delta<0)
    error->all(FLERR,5,"Fix fluid/particle/treatment shell width delta must be zero or positive and finite");
  if (!finite_value(temperature) || temperature<0)
    error->all(FLERR,6,"Fix fluid/particle/treatment temperature T must be zero or positive and finite");
  if (seed==0)
    error->all(FLERR,7,"Fix fluid/particle/treatment seed must not be zero");
  if (!finite_value(maxpenetration) || !(maxpenetration>0 && maxpenetration<1))
    error->all(FLERR,imaxpen,"Fix fluid/particle/treatment maxpenetration {} must lie between 0 and 1 (a fraction of R)",
               maxpenetration);

  // Select once on rank zero, then share the same seed with every rank.
  // This is a convenience seed, not a guarantee of unique independent jobs.
  if (seed<0) {
    if (comm->me==0) {
      const uint64_t ticks=static_cast<uint64_t>(
          std::chrono::system_clock::now().time_since_epoch().count());
      seed=1+static_cast<int>(mix64(ticks)%static_cast<uint64_t>(INT_MAX));
    }
    MPI_Bcast(&seed,1,MPI_INT,0,world);
    if (comm->me==0)
      utils::logmesg(lmp,"Fluid/particle/treatment: time-selected seed = {} (restart may restore saved seed)\n",seed);
  }

  delete[] rigid_id;
  rigid_id=utils::strdup(arg[3]);
  vector_flag=1;
  size_vector=20;
  global_freq=1;
  nevery=1;
  extvector=0; // already-global diagnostics; do not divide by atom count
  force_reneighbor=1;
  next_reneighbor=-1;
  restart_pbc=1;
  restart_global=1;
  no_change_box=1;
  std::fill(stats,stats+20,0.0);
  cumulative_penetrations=cumulative_treatments=0;
}

FixFluidParticleTreatment::~FixFluidParticleTreatment()
{
  if (copymode) return;
  delete[] rigid_id;
}

int FixFluidParticleTreatment::setmask() { return POST_INTEGRATE | END_OF_STEP; }

void FixFluidParticleTreatment::init()
{
  // Exact style names (see style_is()).  This fix needs the plain styles and
  // the KOKKOS subclass one of the three names of each /kk variant.  Any
  // other combination would leave this fix or an integrator working on stale
  // host or device copies of the atoms and of the rigid body.
  const bool kk=kokkos_variant!=0;
  if (strcmp(update->unit_style,"lj") || domain->dimension!=3 ||
      domain->triclinic || !domain->xperiodic || !domain->yperiodic ||
      !domain->zperiodic || !style_is(update->integrate_style,"verlet",kk))
    error->all(FLERR,"Fix {} requires LJ units, run_style {} and a 3D orthogonal fully periodic box{}",
               style,kk ? "verlet/kk" : "verlet",
               kk ? "" : " (with KOKKOS use fix fluid/particle/treatment/kk)");
  if (!atom->tag_enable)
    error->all(FLERR,"Fix {} requires atom IDs",style);

  Fix *rfix=modify->get_fix_by_id(rigid_id);
  if (!rfix || !style_is(rfix->style,"rigid/nve/small",kk))
    error->all(FLERR,"Fix {} requires an existing {} fix ID{}",style,
               kk ? "rigid/nve/small/kk" : "rigid/nve/small",
               kk ? "" : "; its KOKKOS variant rigid/nve/small/kk needs fix fluid/particle/treatment/kk");
  const auto &fixes=modify->get_fix_list();
  const int nfix=static_cast<int>(fixes.size());
  int ir=-1,self=-1;
  for (int f=0;f<nfix;f++) {
    if (fixes[f]==rfix) ir=f;
    if (fixes[f]==this) self=f;
  }
  if (ir>=self)
    error->all(FLERR,"Define fix fluid/particle/treatment after its rigid/nve/small fix");
  // The rigid fix resets the body atom velocities inside this fix's
  // END_OF_STEP (fpt_set_velocities()) in its own KOKKOS execution space, and
  // the atom data is then also marked modified in this fix's space; with two
  // different spaces that is a concurrent host and device modification.
  if (kk && (rfix->execution_space!=execution_space))
    error->all(FLERR,"Fix {} and its rigid fix {} must both run on the device or both on the host; use the same suffix for both (/kk or /kk/device, or /kk/host)",
               style,rigid_id);
  rigid=static_cast<FixRigidSmall *>(rfix);

  // Count by class, not by style name, so a suffixed variant is counted too.
  int nboundary=0;
  for (int f=0;f<nfix;f++) {
    if (dynamic_cast<FixFluidParticleTreatment *>(fixes[f])) nboundary++;
    if (fixes[f]->box_change & (BOX_CHANGE_SIZE | BOX_CHANGE_SHAPE))
      error->all(FLERR,"Fix fluid/particle/treatment does not support a changing box");
  }
  if (nboundary!=1)
    error->all(FLERR,"Only one fix fluid/particle/treatment is supported");

  int invalid=0;
  bigint nfluid=0;
  for (int i=0;i<atom->nlocal;i++) {
    const int fluid_atom=atom->mask[i]&groupbit;
    const int body_atom=atom->mask[i]&rigid->groupbit;
    if (!fluid_atom && !body_atom) continue;
    if (fluid_atom) nfluid++;
    if (fluid_atom && body_atom) invalid=1;
    int integrations=0;
    for (int f=0;f<nfix;f++)
      if (fixes[f]->time_integrate &&
          (atom->mask[i]&fixes[f]->groupbit)) {
        integrations++;
        if (fluid_atom && !style_is(fixes[f]->style,"nve",kk)) invalid=1;
        if (body_atom && fixes[f]!=rigid) invalid=1;
      }
    if (integrations!=1) invalid=1;
  }
  int invalid_all;
  MPI_Allreduce(&invalid,&invalid_all,1,MPI_INT,MPI_MAX,world);
  if (invalid_all)
    error->all(FLERR,"Boundary groups must be disjoint; fluid needs exactly one {} integrator and sphere only its {} integrator",
               kk ? "nve/kk" : "plain nve",kk ? "rigid/nve/small/kk" : "rigid/nve/small");
  bigint totalfluid;
  MPI_Allreduce(&nfluid,&totalfluid,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (!totalfluid) error->all(FLERR,"Fix fluid/particle/treatment fluid group is empty");

  for (int d=0;d<3;d++) {
    box_lengths[d]=domain->prd[d];
    const double extent=std::max(radius+delta,radius*(1+maxpenetration));
    if (!(2*extent<box_lengths[d]))
      error->all(FLERR,"Fluid/particle/treatment radius, shell or relocation reach overlaps periodic images");
  }
}

void FixFluidParticleTreatment::setup(int)
{
  FixRigidSmall::FPTState state;
  rigid->fpt_get_state(state);
  int invalid=0;
  double maxbead=0,minfluid=DBL_MAX;
  scan_setup(state,invalid,maxbead,minfluid);
  int invalid_all;
  double maxbead_all,minfluid_all;
  MPI_Allreduce(&invalid,&invalid_all,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&maxbead,&maxbead_all,1,MPI_DOUBLE,MPI_MAX,world);
  MPI_Allreduce(&minfluid,&minfluid_all,1,MPI_DOUBLE,MPI_MIN,world);
  if (invalid_all)
    error->all(FLERR,"Non-finite coordinates in fix fluid/particle/treatment");
  // Coordinate files written with %.6e can displace boundary sites by ~1e-7 R.
  if (maxbead_all>radius*(1+1e-6))
    error->all(FLERR,"Rigid beads lie outside the specified COM-centered sphere; verify R and geometric center");
  if (minfluid_all<radius*(1-1e-12))
    error->all(FLERR,"Initial solvent lies inside the sphere; generate an excluded-volume initial configuration");
  std::fill(stats,stats+20,0.0);
  stats[15]=minfluid_all;
  stats[16]=cumulative_penetrations;
  stats[17]=cumulative_treatments;
  next_reneighbor=-1;
  penetrated_tags.clear();
  if (comm->me==0) {
    const char *mode=thermal ? "thermal" : "penetration";
    utils::logmesg(lmp,"Fluid/particle/treatment: mode {}, R {:.8g}, delta {:.8g}, T {:.8g}; geometry centered on rigid COM\n",mode,radius,delta,temperature);
  }
}

// Distances of all owned solvent and sphere atoms from the body COM.
void FixFluidParticleTreatment::scan_setup(const FixRigidSmall::FPTState &state,
                                           int &invalid,double &maxbead,double &minfluid)
{
  for (int i=0;i<atom->nlocal;i++) {
    const int mask=atom->mask[i];
    if (!(mask&(groupbit|rigid->groupbit))) continue;
    double r[3];
    for (int d=0;d<3;d++) r[d]=atom->x[i][d]-state.xcm[d];
    if (!finite_value(r[0]) || !finite_value(r[1]) || !finite_value(r[2])) {
      invalid=1;
      continue;
    }
    minimum_image_unbounded(domain->periodicity,domain->prd,domain->prd_half,r);
    const double dist=std::sqrt(dot(r,r));
    if (!finite_value(dist)) invalid=1;
    if (mask&rigid->groupbit) maxbead=std::max(maxbead,dist);
    if (mask&groupbit) minfluid=std::min(minfluid,dist);
  }
}

void FixFluidParticleTreatment::post_integrate()
{
  std::fill(stats,stats+20,0.0);
  next_reneighbor=-1;
  penetrated_tags.clear();
  process_boundary(0);
}

void FixFluidParticleTreatment::end_of_step()
{
  if (thermal) process_boundary(1);
}

// Phase 0: radial projection after drift. Phase 1: velocity heatbath after
// every integrator's final force kick. Recollect owned atoms by tag in phase
// 1 because projection can migrate atoms and rebuild/sort neighbor lists.
void FixFluidParticleTreatment::process_boundary(int thermal_phase)
{
  for (int d=0;d<3;d++) if (domain->prd[d]!=box_lengths[d])
    error->all(FLERR,"Fix fluid/particle/treatment does not support changing box dimensions");

  FixRigidSmall::FPTState state;
  rigid->fpt_get_state(state);

  std::vector<Contact> local;
  int invalid=0;
  double minradius=DBL_MAX;
  collect_contacts(thermal_phase,state,local,invalid,minradius);
  int invalid_all;
  MPI_Allreduce(&invalid,&invalid_all,1,MPI_INT,MPI_MAX,world);
  if (invalid_all)
    error->all(FLERR,"Invalid fluid mass, ID, coordinates or velocities in fluid/particle/treatment");
  MPI_Allreduce(&minradius,&stats[15],1,MPI_DOUBLE,MPI_MIN,world);

  // Only owned near-boundary solvent is collected. Each rank processes the
  // same globally ID-sorted stream; every event sees the updated body U,L.
  if (local.size()>static_cast<size_t>(INT_MAX/sizeof(Contact)))
    error->one(FLERR,"Too many local fluid/particle/treatment contacts");
  const int localbytes=static_cast<int>(local.size()*sizeof(Contact));
  std::vector<int> counts(comm->nprocs),displs(comm->nprocs);
  MPI_Allgather(const_cast<int *>(&localbytes),1,MPI_INT,&counts[0],1,MPI_INT,world);
  int totalbytes=0;
  for (int p=0;p<comm->nprocs;p++) {
    displs[p]=totalbytes;
    if (counts[p]>INT_MAX-totalbytes)
      error->all(FLERR,"Too many global fluid/particle/treatment contacts");
    totalbytes+=counts[p];
  }
  const size_t ncontacts=totalbytes/sizeof(Contact);
  std::vector<Contact> contacts(ncontacts);
  Contact dummy;
  MPI_Allgatherv(local.empty() ? &dummy : &local[0],localbytes,MPI_BYTE,
                 contacts.empty() ? &dummy : &contacts[0],&counts[0],
                 &displs[0],MPI_BYTE,world);
  std::sort(contacts.begin(),contacts.end(),contact_less);
  stats[19]=static_cast<double>(ncontacts);

  double linear_residual[3]={0,0,0},angular_residual[3]={0,0,0};
  for (size_t n=0;n<ncontacts;n++) {
    Contact &c=contacts[n];
    if (n && c.tag==contacts[n-1].tag)
      error->all(FLERR,"Duplicate solvent atom ID in fluid/particle/treatment");
    const double dist=std::sqrt(dot(c.r,c.r));
    if (!thermal_phase && dist<radius) {
      penetrated_tags.push_back(c.tag);
      const double depth=radius-dist;
      stats[0]+=1;
      stats[2]=std::max(stats[2],depth);
      if (!(dist>1e-12*radius) || depth>maxpenetration*radius)
        error->all(FLERR,"Fluid/particle/treatment atom {} penetration {:.8g} exceeds permitted {:.8g}; reduce timestep/check initial state",
                   c.tag,depth,maxpenetration*radius);
      // Reflect penetration depth radially. This changes configuration,
      // not momentum. No lab-velocity-dependent compensating torque is
      // applied: such compensation would violate Galilean invariance.
      double orbital_defect[3],momentum[3];
      for (int d=0;d<3;d++) {
        c.shift[d]=(2*depth/dist)*c.r[d];
        momentum[d]=c.mass*c.v[d];
        c.r[d]+=c.shift[d];
      }
      cross(c.shift,momentum,orbital_defect);
      for (int d=0;d<3;d++) stats[9+d]+=orbital_defect[d];
      stats[18]=std::max(stats[18],2*depth);
    }

    if (thermal_phase) {
      double oldU[3],oldL[3],oldv[3],z[3],J[3];
      for (int d=0;d<3;d++) {
        oldU[d]=state.vcm[d]; oldL[d]=state.angmom[d]; oldv[d]=c.v[d];
      }
      const double before=pair_ke(c,state);
      gaussian3(seed,update->ntimestep,c.tag,z);
      if (!heatbath(c.mass,state.mass,state.inverse_inertia,c.r,temperature,z,
                    c.v,state.vcm,state.angmom,J))
        error->all(FLERR,"Non-positive collision matrix in fluid/particle/treatment");
      stats[1]+=1;
      stats[14]+=pair_ke(c,state)-before;
      double dp[3],orbital[3];
      for (int d=0;d<3;d++) {
        dp[d]=c.mass*(c.v[d]-oldv[d]);
        const double dP=state.mass*(state.vcm[d]-oldU[d]);
        const double dL=state.angmom[d]-oldL[d];
        stats[3+d]+=dP;
        stats[6+d]+=dL;
        linear_residual[d]+=dP+dp[d];
        angular_residual[d]+=dL;
      }
      cross(c.r,dp,orbital);
      for (int d=0;d<3;d++) angular_residual[d]+=orbital[d];
    }
  }

  if (thermal_phase) {
    stats[12]=std::sqrt(dot(linear_residual,linear_residual));
    stats[13]=std::sqrt(dot(angular_residual,angular_residual));
    cumulative_treatments+=stats[1];
    // Every tag saved before migration must occur once in the new contact
    // list. A deletion/group mutation between phases is unsupported.
    size_t found=0;
    for (size_t n=0;n<ncontacts;n++)
      if (std::binary_search(penetrated_tags.begin(),penetrated_tags.end(),contacts[n].tag))
        found++;
    if (found!=penetrated_tags.size())
      error->all(FLERR,"A penetrated solvent atom disappeared before end-of-step boundary treatment");
  } else cumulative_penetrations+=stats[0];
  stats[16]=cumulative_penetrations;
  stats[17]=cumulative_treatments;

  apply_contacts(thermal_phase,contacts);
  if (thermal_phase && ncontacts)
    rigid->fpt_set_velocities(state.vcm,state.angmom);
  // Neighbor::decide checks next_reneighbor before its usual every/delay
  // rules. A projection must migrate atoms/update ghosts/rebuild lists now.
  if (!thermal_phase && stats[0]>0) next_reneighbor=update->ntimestep;
}

// Phase 0: owned solvent inside the sphere.  Phase 1: owned solvent in the
// thermal shell, or penetrated in phase 0 of this step.  Also flags invalid
// atoms and returns the smallest (projected) distance from the COM.
void FixFluidParticleTreatment::collect_contacts(int thermal_phase,
    const FixRigidSmall::FPTState &state,std::vector<Contact> &local,
    int &invalid,double &minradius)
{
  const double shell_radius=radius+delta;
  for (int i=0;i<atom->nlocal;i++) if (atom->mask[i]&groupbit) {
    Contact c;
    std::memset(&c,0,sizeof(c));
    c.tag=atom->tag[i]; c.owner=comm->me; c.index=i;
    c.mass=atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
    for (int d=0;d<3;d++) {
      c.r[d]=atom->x[i][d]-state.xcm[d];
      c.v[d]=atom->v[i][d];
      if (!finite_value(c.r[d]) || !finite_value(c.v[d])) invalid=1;
    }
    // The minimum-image loop never ends for an infinite separation; skip
    // such atoms so that the collective error below can be raised.
    if (!finite_value(c.r[0]) || !finite_value(c.r[1]) || !finite_value(c.r[2]))
      continue;
    minimum_image_unbounded(domain->periodicity,domain->prd,domain->prd_half,c.r);
    if (!finite_value(c.mass) || !(c.mass>0) || c.tag<=0) invalid=1;
    const double dist=std::sqrt(dot(c.r,c.r));
    if (!finite_value(dist)) invalid=1;
    // Thermal phase uses current full-step velocities and actual positions.
    const double finaldist=(!thermal_phase && dist<radius) ? 2*radius-dist : dist;
    minradius=std::min(minradius,finaldist);
    if (thermal_phase) {
      if (dist<radius*(1-1e-12)) invalid=1;
      // A deeply reflected bead can lie outside the thermal shell. It must
      // still receive one thermal boundary collision during this step.
      if (dist<shell_radius ||
          std::binary_search(penetrated_tags.begin(),penetrated_tags.end(),c.tag))
        local.push_back(c);
    } else if (dist<radius) local.push_back(c);
  }
}

// No exchange or atom sorting has happened since contacts were collected,
// so the original owning rank/local index is still valid.
void FixFluidParticleTreatment::apply_contacts(int thermal_phase,
                                               const std::vector<Contact> &contacts)
{
  const size_t ncontacts=contacts.size();
  for (size_t n=0;n<ncontacts;n++) {
    const Contact &c=contacts[n];
    if (c.owner!=comm->me) continue;
    if (atom->tag[c.index]!=c.tag)
      error->one(FLERR,"Contact ownership changed during fluid/particle/treatment");
    for (int d=0;d<3;d++)
      if (thermal_phase) atom->v[c.index][d]=c.v[d];
      else atom->x[c.index][d]+=c.shift[d];
    // Preserve unwrapped trajectories by updating image counters when the
    // corrected position crosses a periodic face.
    if (dot(c.shift,c.shift)>0) domain->remap(atom->x[c.index],atom->image[c.index]);
  }
}

double FixFluidParticleTreatment::compute_vector(int n) { return stats[n]; }

void FixFluidParticleTreatment::wall_geometry(double *center,double &r,double &shell,
                                              int wrapped)
{
  if (!rigid) error->all(FLERR,"Sphere boundary must initialize before wall boundary");
  FixRigidSmall::FPTState state;
  rigid->fpt_get_state(state);
  for (int d=0;d<3;d++) center[d]=wrapped ? state.xcm_wrapped[d] : state.xcm[d];
  r=radius;
  shell=thermal ? delta : 0.0;
}

void FixFluidParticleTreatment::write_restart(FILE *fp)
{
  if (comm->me==0) {
    // Negative version marker distinguishes this from the legacy two counters.
    const double data[4]={-1.0,cumulative_penetrations,cumulative_treatments,
                          static_cast<double>(seed)};
    int bytes=sizeof(data);
    fwrite(&bytes,sizeof(int),1,fp);
    fwrite(data,sizeof(double),4,fp);
  }
}

void FixFluidParticleTreatment::restart(char *buffer)
{
  double marker;
  memcpy(&marker,buffer,sizeof(double));
  if (marker==-1.0) {
    double data[4];
    memcpy(data,buffer,sizeof(data));
    cumulative_penetrations=data[1];
    cumulative_treatments=data[2];
    seed=static_cast<int>(data[3]);
    if (comm->me==0)
      utils::logmesg(lmp,"Fluid/particle/treatment: restored seed = {}\n",seed);
  } else {
    double data[2];
    memcpy(data,buffer,sizeof(data));
    cumulative_penetrations=data[0];
    cumulative_treatments=data[1];
    if (comm->me==0)
      error->warning(FLERR,"Legacy fluid/particle/treatment restart has no seed; use the original positive seed for reproducibility");
  }
}
