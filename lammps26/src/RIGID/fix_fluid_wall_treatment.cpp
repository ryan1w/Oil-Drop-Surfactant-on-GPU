// clang-format off
/* LAMMPS stationary planar solvent boundary, 2026.
   Distributed under the GNU General Public License, version 2 or later.

   Li & Drazer (2008), Sec. II B: specular reflection of penetrating
   solvent and a Gaussian heatbath in a thin fluid-side surface layer.
   This is our implementation, not the authors' original code.

   fix ID fluid fluid/wall/treatment wallGroup zlo zhi delta T seed
       [mode thermal|reflect] [maxpenetration distance]

   LJ units, fixed orthogonal 3D box, lateral periodicity, plain solvent
   NVE and stationary wall beads. Explicit walls must lie inside the
   box; z may be periodic or fixed. Reflection occurs after drift and
   the heatbath AFTER the final Verlet kick. All three thermal velocity
   components have mean zero and variance T/m (no outward-only bias).
   A bead reflected past the layer is not additionally thermalized.
   This fix requires the plain verlet and nve styles; with KOKKOS use
   fix fluid/wall/treatment/kk (src/KOKKOS), which requires their /kk
   variants and replaces the per-atom loops (reflect_atoms, scan_atoms,
   thermalize_atoms) with device kernels.

   If the sphere fix exists, define it before this fix, on the same fluid
   group. Its read-only geometry query prevents unresolved sphere/wall
   overlap and identifies reflections that would put fluid in the sphere.
 */
#include "fix_fluid_wall_treatment.h"
#include "fix_fluid_particle_treatment.h"
#include "fluid_particle_treatment_math.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "modify.h"
#include "update.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
namespace {
bool finite_value(double a) { return a==a && std::fabs(a)<=DBL_MAX; }
uint64_t mix64(uint64_t x)
{
  x+=UINT64_C(0x9e3779b97f4a7c15);
  x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
  return x^(x>>31);
}
double uniform_open(uint64_t key,uint64_t lane)
{
  return (static_cast<double>(mix64(key+UINT64_C(0x9e3779b97f4a7c15)*lane)>>12)+0.5)
      /4503599627370496.0;
}
void gaussian3(int seed,bigint step,tagint tag,int side,double *g)
{
  // Separate RNG namespace from the sphere fix; independent of MPI rank.
  uint64_t key=mix64(static_cast<uint64_t>(seed)^UINT64_C(0x57414c4c48454154));
  key=mix64(key^static_cast<uint64_t>(step));
  key=mix64(key^static_cast<uint64_t>(tag));
  key=mix64(key^static_cast<uint64_t>(side));
  const double twopi=6.283185307179586476925286766559;
  double a=std::sqrt(-2*std::log(uniform_open(key,1)));
  double p=twopi*uniform_open(key,2);
  g[0]=a*std::cos(p); g[1]=a*std::sin(p);
  a=std::sqrt(-2*std::log(uniform_open(key,3)));
  p=twopi*uniform_open(key,4);
  g[2]=a*std::cos(p);
}
}

FixFluidWallTreatment::FixFluidWallTreatment(LAMMPS *lmp,int narg,char **arg)
  : Fix(lmp,narg,arg),wall_id(nullptr),kokkos_variant(0),sphere(nullptr)
{
  if (narg<9) utils::missing_cmd_args(FLERR,"fix fluid/wall/treatment",error);
  delete[] wall_id;
  wall_id=utils::strdup(arg[3]);
  lo=utils::numeric(FLERR,arg[4],false,lmp); hi=utils::numeric(FLERR,arg[5],false,lmp);
  delta=utils::numeric(FLERR,arg[6],false,lmp); temperature=utils::numeric(FLERR,arg[7],false,lmp);
  seed=utils::inumeric(FLERR,arg[8],false,lmp);
  thermal=1; maxpenetration=0.25;
  int imaxpen=Error::NOPOINTER;   // argument index of maxpenetration value
  for (int i=9;i<narg;) {
    if (!strcmp(arg[i],"mode")) {
      if (i+1>=narg) utils::missing_cmd_args(FLERR,"fix fluid/wall/treatment mode",error);
      if (!strcmp(arg[i+1],"thermal")) thermal=1;
      else if (!strcmp(arg[i+1],"reflect")) thermal=0;
      else error->all(FLERR,i+1,"Fluid/wall/treatment mode must be thermal or reflect");
      i+=2;
    } else if (!strcmp(arg[i],"maxpenetration")) {
      if (i+1>=narg) utils::missing_cmd_args(FLERR,"fix fluid/wall/treatment maxpenetration",error);
      maxpenetration=utils::numeric(FLERR,arg[i+1],false,lmp); imaxpen=i+1; i+=2;
    } else error->all(FLERR,i,"Unknown fix fluid/wall/treatment keyword: {}",arg[i]);
  }
  // Same conditions as the single combined check of the 2018 version.
  if (!finite_value(lo))
    error->all(FLERR,4,"Fix fluid/wall/treatment zlo must be finite");
  if (!finite_value(hi) || !(hi>lo))
    error->all(FLERR,5,"Fix fluid/wall/treatment zhi must be finite and larger than zlo");
  if (!finite_value(delta) || delta<0 || !(2*delta<hi-lo))
    error->all(FLERR,6,"Fix fluid/wall/treatment layer width delta must be zero or positive and less than (zhi-zlo)/2");
  if (!finite_value(temperature) || temperature<0)
    error->all(FLERR,7,"Fix fluid/wall/treatment temperature T must be zero or positive and finite");
  if (seed==0)
    error->all(FLERR,8,"Fix fluid/wall/treatment seed must not be zero");
  if (!finite_value(maxpenetration) || !(maxpenetration>0 && maxpenetration<hi-lo))
    error->all(FLERR,imaxpen,"Fix fluid/wall/treatment maxpenetration {} must be positive and less than zhi-zlo",
               maxpenetration);
  if (seed<0) {
    if (comm->me==0) {
      const uint64_t ticks=static_cast<uint64_t>(
          std::chrono::system_clock::now().time_since_epoch().count());
      seed=1+static_cast<int>(mix64(ticks)%static_cast<uint64_t>(INT_MAX));
    }
    MPI_Bcast(&seed,1,MPI_INT,0,world);
  }
  vector_flag=1; size_vector=20; global_freq=1; nevery=1; extvector=0;
  restart_global=1; restart_pbc=1;
  no_change_box=1; time_depend=1;
  std::fill(stats,stats+20,0.0); std::fill(cumulative,cumulative+4,0.0);
}
FixFluidWallTreatment::~FixFluidWallTreatment()
{
  if (copymode) return;
  delete[] wall_id;
}
int FixFluidWallTreatment::setmask() { return POST_INTEGRATE | END_OF_STEP; }

void FixFluidWallTreatment::init()
{
  // Exact style names (see FluidParticleMath::style_is()): plain verlet and
  // nve for this fix, one of the three names of their /kk variants for the
  // KOKKOS subclass.
  const bool kk=kokkos_variant!=0;
  if (strcmp(update->unit_style,"lj") || domain->dimension!=3 || domain->triclinic ||
      !domain->xperiodic || !domain->yperiodic ||
      !FluidParticleMath::style_is(update->integrate_style,"verlet",kk))
    error->all(FLERR,"Fix {} requires LJ units, run_style {}, orthogonal 3D box and periodic x,y{}",
               style,kk ? "verlet/kk" : "verlet",
               kk ? "" : " (with KOKKOS use fix fluid/wall/treatment/kk)");
  if ((!domain->zperiodic && (domain->boundary[2][0]!=1 || domain->boundary[2][1]!=1)) ||
      !(domain->boxlo[2]<lo && hi<domain->boxhi[2]))
    error->all(FLERR,"Wall planes must be strictly inside a periodic or fixed z box");
  if (!atom->tag_enable || !comm->ghost_velocity)
    error->all(FLERR,"Fluid/wall/treatment requires atom IDs and comm_modify vel yes");
  const int iw=group->find(wall_id);
  if (iw<0) error->all(FLERR,"Fluid/wall/treatment wall group does not exist");
  if (group->dynamic[iw]) error->all(FLERR,"Fluid/wall/treatment requires a static wall group");
  wallbit=group->bitmask[iw];
  const auto &fixes=modify->get_fix_list();
  const int nfix=static_cast<int>(fixes.size());
  int self=-1;
  for (int f=0;f<nfix;f++) if (fixes[f]==this) self=f;
  // Identify both treatment fixes by class, not by style name: a suffixed
  // variant of the sphere fix must not be missed silently, because running
  // without a sphere fix is also a valid setup and would raise no error.
  int count=0;
  sphere=nullptr;
  for (int f=0;f<nfix;f++) {
    Fix *other=fixes[f];
    if (dynamic_cast<FixFluidWallTreatment *>(other)) count++;
    if (other->box_change & (BOX_CHANGE_SIZE | BOX_CHANGE_SHAPE))
      error->all(FLERR,"Fluid/wall/treatment does not support a changing box");
    auto *candidate=dynamic_cast<FixFluidParticleTreatment *>(other);
    if (candidate) {
      if (sphere || f>=self || other->igroup!=igroup)
        error->all(FLERR,"Define one sphere treatment before wall treatment on the same fluid group");
      sphere=candidate;
    }
  }
  if (count!=1) error->all(FLERR,"Only one fluid/wall/treatment fix is supported");
  int invalid=0;
  bigint counts[3]={0,0,0},total[3];
  for (int i=0;i<atom->nlocal;i++) {
    const bool fluid=(atom->mask[i]&groupbit)!=0;
    const bool wall=(atom->mask[i]&wallbit)!=0;
    if (!fluid && !wall) continue;
    if (fluid && wall) invalid=1;
    if (fluid) counts[0]++;
    if (wall) counts[atom->x[i][2]<=lo ? 1 : 2]++;
    int integrations=0;
    for (int f=0;f<nfix;f++) {
      Fix *other=fixes[f];
      if (!(atom->mask[i]&other->groupbit)) continue;
      if (!strcmp(other->style,"move")) invalid=1;
      if (other->time_integrate) {
        integrations++;
        if (!FluidParticleMath::style_is(other->style,"nve",kk)) invalid=1;
      }
    }
    if ((fluid && integrations!=1) || (wall && integrations!=0)) invalid=1;
  }
  int bad;
  MPI_Allreduce(&invalid,&bad,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(counts,total,3,MPI_LMP_BIGINT,MPI_SUM,world);
  if (bad) error->all(FLERR,"Wall/fluid groups must be disjoint; walls unintegrated and solvent integrated once by {}",
                      kk ? "nve/kk" : "plain nve");
  if (!total[0] || !total[1] || !total[2])
    error->all(FLERR,"Fluid/wall/treatment requires solvent and beads in both walls");
  for (int d=0;d<3;d++) { bounds[d]=domain->boxlo[d]; bounds[d+3]=domain->boxhi[d]; }
}

void FixFluidWallTreatment::geometry(double *center,double &r)
{
  for (int d=0;d<3;d++)
    if (bounds[d]!=domain->boxlo[d] || bounds[d+3]!=domain->boxhi[d])
      error->all(FLERR,"Fluid/wall/treatment box bounds changed");
  r=0;
  if (sphere) {
    double shell;
    // collective, called by every rank; the KOKKOS subclass measures from the
    // wrapped COM with an O(1) minimum image
    sphere->wall_geometry(center,r,shell,kokkos_variant);
    const double gap=std::min(center[2]-lo-r,hi-center[2]-r);
    if (!(gap>0)) error->all(FLERR,"Sphere touches or crosses a wall treatment plane");
    if (gap+1e-12*std::max(1.0,r)<shell+(thermal ? delta : 0.0))
      error->all(FLERR,"Sphere and wall thermal layers overlap; this regime is not supported");
  }
}

void FixFluidWallTreatment::check_atoms(int initial)
{
  double center[3]={0,0,0},r;
  geometry(center,r);
  int bad=0;
  double gaps[2]={DBL_MAX,DBL_MAX},global[2];
  scan_atoms(center,r,bad,gaps);
  int allbad;
  MPI_Allreduce(&bad,&allbad,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(gaps,global,2,MPI_DOUBLE,MPI_MIN,world);
  if (allbad) error->all(FLERR,initial ?
      "Invalid wall setup: solvent outside accessible fluid, nonstationary walls, invalid mass or nonfinite state" :
      "Invalid state after wall correction: solvent in a solid or nonstationary wall; reduce timestep/check geometry");
  stats[13]=global[0]; stats[14]=global[1];
}

// Validity of all owned solvent and wall atoms; smallest solvent-wall gaps.
void FixFluidWallTreatment::scan_atoms(const double *center,double r,int &bad,double *gaps)
{
  for (int i=0;i<atom->nlocal;i++) {
    const int mask=atom->mask[i];
    if (!(mask&(groupbit|wallbit))) continue;
    const double z=atom->x[i][2];
    for (int d=0;d<3;d++)
      if (!finite_value(atom->x[i][d]) || !finite_value(atom->v[i][d])) bad=1;
    if (mask&wallbit) {
      if (z>lo+1e-8 && z<hi-1e-8) bad=1;
      for (int d=0;d<3;d++) if (std::fabs(atom->v[i][d])>1e-14) bad=1;
    }
    if (!(mask&groupbit)) continue;
    if (z<lo || z>hi) bad=1;
    const double m=atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
    if (!(m>0) || !finite_value(m) || atom->tag[i]<=0) bad=1;
    gaps[0]=std::min(gaps[0],z-lo); gaps[1]=std::min(gaps[1],hi-z);
    if (sphere) {
      double dr[3];
      bool good=true;
      for (int d=0;d<3;d++) { dr[d]=atom->x[i][d]-center[d]; good=good && finite_value(dr[d]); }
      if (!good) { bad=1; continue; }
      // center is the UNWRAPPED sphere COM: see minimum_image_unbounded()
      FluidParticleMath::minimum_image_unbounded(domain->periodicity,domain->prd,
                                                 domain->prd_half,dr);
      if (dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2]<r*r*(1-2e-12)) bad=1;
    }
  }
}

void FixFluidWallTreatment::setup(int)
{
  std::fill(stats,stats+20,0.0);
  check_atoms(1);
  for (int j=0;j<4;j++) stats[15+j]=cumulative[j];
  if (comm->me==0)
    utils::logmesg(lmp,"Fluid/wall/treatment: zlo {:.12g}, zhi {:.12g}, delta {:.8g}, T {:.8g}, seed {}, mode {}\n",lo,hi,delta,temperature,seed,thermal ? "thermal" : "reflect");
}

void FixFluidWallTreatment::post_integrate()
{
  std::fill(stats,stats+20,0.0);
  double local[12]={0},global[12],maxima[3]={0},maxall[3];
  int bad=0;
  reflect_atoms(local,maxima,bad);
  int allbad;
  MPI_Allreduce(&bad,&allbad,1,MPI_INT,MPI_MAX,world);
  if (allbad) error->all(FLERR,"Excessive wall penetration or nonfinite position; reduce timestep");
  MPI_Allreduce(local,global,12,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(maxima,maxall,3,MPI_DOUBLE,MPI_MAX,world);
  stats[0]=global[0]; stats[1]=global[1];
  stats[4]=maxall[0]; stats[5]=maxall[1]; stats[19]=maxall[2];
  for (int j=6;j<12;j++) stats[j]=global[j];
  cumulative[0]+=stats[0]; cumulative[1]+=stats[1];
  check_atoms(0); // catches a planar reflection into the sphere
  // No forced reneighbor: Neighbor::decide() runs after post_integrate and
  // its skin check sees the reflected positions, which stay inside the box.
}

void FixFluidWallTreatment::end_of_step()
{
  check_atoms(0); // recollect OWNED atoms after migration and sphere heatbath
  double local[9]={0},global[9];
  if (thermal) thermalize_atoms(local);
  MPI_Allreduce(local,global,9,MPI_DOUBLE,MPI_SUM,world);
  stats[2]=global[0]; stats[3]=global[1]; stats[12]=global[8];
  for (int j=0;j<6;j++) stats[6+j]+=global[2+j];
  cumulative[2]+=stats[2]; cumulative[3]+=stats[3];
  for (int j=0;j<4;j++) stats[15+j]=cumulative[j];
}

// Specular reflection of owned solvent that crossed a wall plane.
void FixFluidWallTreatment::reflect_atoms(double *local,double *maxima,int &bad)
{
  for (int i=0;i<atom->nlocal;i++) if (atom->mask[i]&groupbit) {
    const double z=atom->x[i][2];
    if (!finite_value(z)) { bad=1; continue; }
    const int side=z<lo ? 0 : (z>hi ? 1 : -1);
    if (side<0) continue;
    const double depth=side==0 ? lo-z : z-hi;
    if (depth>maxpenetration) { bad=1; continue; }
    const double m=atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
    const double oldv=atom->v[i][2];
    local[side]++;
    // Impulse received by the fixed wall, opposite to the solvent change.
    local[6+3*side+2]+=2*m*oldv;
    maxima[side]=std::max(maxima[side],depth);
    maxima[2]=std::max(maxima[2],2*depth);
    atom->x[i][2]=(side==0) ? lo+depth : hi-depth;
    atom->v[i][2]=-oldv;
  }
}

// Gaussian velocity reset of owned solvent in the two thermal layers.
void FixFluidWallTreatment::thermalize_atoms(double *local)
{
  for (int i=0;i<atom->nlocal;i++) if (atom->mask[i]&groupbit) {
    const double z=atom->x[i][2];
    const int side=(z-lo<delta) ? 0 : ((hi-z<delta) ? 1 : -1);
    if (side<0) continue;
    const double m=atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
    double g[3]; gaussian3(seed,update->ntimestep,atom->tag[i],side,g);
    const double sigma=std::sqrt(temperature/m);
    local[side]++;
    for (int d=0;d<3;d++) {
      const double oldv=atom->v[i][d],newv=sigma*g[d];
      local[2+3*side+d]+=m*(oldv-newv);
      local[8]+=0.5*m*(newv*newv-oldv*oldv);
      atom->v[i][d]=newv;
    }
  }
}

double FixFluidWallTreatment::compute_vector(int n) { return stats[n]; }

void FixFluidWallTreatment::write_restart(FILE *fp)
{
  if (comm->me!=0) return;
  const double data[13]={1,static_cast<double>(seed),lo,hi,delta,temperature,
      maxpenetration,static_cast<double>(thermal),cumulative[0],cumulative[1],
      cumulative[2],cumulative[3],0};
  const int bytes=sizeof(data);
  fwrite(&bytes,sizeof(int),1,fp); fwrite(data,sizeof(double),13,fp);
}
void FixFluidWallTreatment::restart(char *buffer)
{
  double data[13]; std::memcpy(data,buffer,sizeof(data));
  if (data[0]!=1 || data[2]!=lo || data[3]!=hi || data[4]!=delta ||
      data[5]!=temperature || data[6]!=maxpenetration || data[7]!=thermal)
    error->all(FLERR,"Fluid/wall/treatment restart settings do not match saved fix");
  seed=static_cast<int>(data[1]);
  for (int j=0;j<4;j++) cumulative[j]=data[8+j];
}
