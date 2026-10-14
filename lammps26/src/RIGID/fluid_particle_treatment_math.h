/* Exact fixed-position, finite-mass velocity heatbath used by
   fix fluid/particle/treatment, and the minimum-image helper shared with
   fix fluid/wall/treatment.  Independent of LAMMPS for unit tests.
   This file is distributed under the GNU General Public License. */
#ifndef LMP_FLUID_PARTICLE_TREATMENT_MATH_H
#define LMP_FLUID_PARTICLE_TREATMENT_MATH_H

#include <cmath>
#include <string>

namespace LAMMPS_NS {
namespace FluidParticleMath {

// Exact style-name test used by the treatment fixes.  kokkos = false: the
// style must be exactly base.  kokkos = true: it must be one of the three
// names of the KOKKOS variant, base/kk, base/kk/device or base/kk/host.  No
// prefix or suffix stripping: e.g. nve/sphere or rigid/nve/small/omp are
// different integrators.
inline bool style_is(const char *style, const char *base, bool kokkos)
{
  if (!style || !base) return false;
  const std::string s(style), b(base);
  if (!kokkos) return s == b;
  return (s == b + "/kk") || (s == b + "/kk/device") || (s == b + "/kk/host");
}

inline double dot(const double *a, const double *b)
{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

inline void cross(const double *a, const double *b, double *c)
{
  const double x=a[1]*b[2]-a[2]*b[1];
  const double y=a[2]*b[0]-a[0]*b[2];
  const double z=a[0]*b[1]-a[1]*b[0];
  c[0]=x; c[1]=y; c[2]=z;
}

// Minimum image of a separation in an orthogonal box, shifting by one
// period at a time: the arithmetic of Domain::minimum_image(double *) in
// LAMMPS 2018, so results are bit-identical to it.  Unlike the current
// Domain::minimum_image(), there is no limit on the number of periods: the
// fixes measure from the UNWRAPPED body COM, which moves one box length
// further away at every periodic crossing of the body.  The loop takes one
// pass per crossing, and an infinite component would never leave it, so
// callers must pass finite values only.  periodic, prd and half are
// Domain::periodicity, Domain::prd and Domain::prd_half.  (The KOKKOS
// variants measure from the wrapped COM and use an O(1) minimum image.)
inline void minimum_image_unbounded(const int *periodic, const double *prd,
                                    const double *half, double *delta)
{
  for (int d=0;d<3;d++) {
    if (!periodic[d]) continue;
    while (std::fabs(delta[d])>half[d]) {
      if (delta[d]<0.0) delta[d]+=prd[d];
      else delta[d]-=prd[d];
    }
  }
}

inline void matvec(const double a[3][3], const double *x, double *y)
{
  double z[3];
  for (int i=0;i<3;i++) z[i]=dot(a[i],x);
  for (int i=0;i<3;i++) y[i]=z[i];
}

inline double rotational_energy(const double invI[3][3], const double *L)
{
  double omega[3];
  matvec(invI,L,omega);
  return 0.5*dot(L,omega);
}

// K = (1/m + 1/M) identity - [r]x invI [r]x, where [r]x x = r cross x.
// A solvent impulse J changes c=v-U-omega cross r by K J.
inline void effective_matrix(double m, double M, const double invI[3][3],
                             const double *r, double K[3][3])
{
  const double S[3][3]={{0,-r[2],r[1]}, {r[2],0,-r[0]},
                         {-r[1],r[0],0}};
  double temp[3][3];
  for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) {
      temp[i][j]=0;
      for (int k=0;k<3;k++) temp[i][j]+=invI[i][k]*S[k][j];
    }
  for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) {
      K[i][j]=(i==j ? 1.0/m+1.0/M : 0.0);
      for (int k=0;k<3;k++) K[i][j]-=S[i][k]*temp[k][j];
    }
}

inline bool cholesky(const double K[3][3], double B[3][3])
{
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) B[i][j]=0;
  for (int i=0;i<3;i++) {
    for (int j=0;j<=i;j++) {
      double x=K[i][j];
      for (int k=0;k<j;k++) x-=B[i][k]*B[j][k];
      if (i==j) {
        if (!(x>0.0)) return false;
        B[i][j]=std::sqrt(x);
      } else B[i][j]=x/B[j][j];
    }
  }
  return true;
}

inline void solve_cholesky(const double B[3][3], const double *rhs, double *x)
{
  double y[3];
  for (int i=0;i<3;i++) {
    y[i]=rhs[i];
    for (int j=0;j<i;j++) y[i]-=B[i][j]*y[j];
    y[i]/=B[i][i];
  }
  for (int i=2;i>=0;i--) {
    x[i]=y[i];
    for (int j=i+1;j<3;j++) x[i]-=B[j][i]*x[j];
    x[i]/=B[i][i];
  }
}

// z contains three independent unit Gaussian variates.
// Draw c_new from N(0,T K) and apply J=K^-1(c_new-c).
// This is a conditional Maxwell heatbath at FIXED POSITIONS, preserving
// pair linear and angular momentum. It does not make radial relocation
// an exact hard-sphere configurational sampler.
inline bool heatbath(double m, double M, const double invI[3][3],
                     const double *r, double T, const double *z,
                     double *v, double *U, double *L, double *J)
{
  double K[3][3],B[3][3],omega[3],rotv[3],rhs[3];
  effective_matrix(m,M,invI,r,K);
  if (!cholesky(K,B)) return false;
  matvec(invI,L,omega);
  cross(omega,r,rotv);
  for (int i=0;i<3;i++) {
    double target=0;
    for (int j=0;j<=i;j++) target+=B[i][j]*z[j];
    rhs[i]=std::sqrt(T)*target-(v[i]-U[i]-rotv[i]);
  }
  solve_cholesky(B,rhs,J);
  double torque[3];
  cross(r,J,torque);
  for (int i=0;i<3;i++) {
    v[i]+=J[i]/m;
    U[i]-=J[i]/M;
    L[i]-=torque[i];
  }
  return true;
}

} // namespace FluidParticleMath
} // namespace LAMMPS_NS
#endif
