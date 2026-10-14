/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Independent conservative and dissipative cutoffs.
   Original style by Zhen Li (Clemson University); ported to current LAMMPS.
------------------------------------------------------------------------- */

#include "pair_dpd_diffcut.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "random_mars.h"
#include "update.h"

#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;

static constexpr double EPSILON = 1.0e-10;

PairDPDDiffCut::PairDPDDiffCut(LAMMPS *lmp) :
    PairDPD(lmp), cut_c(nullptr), cut_d(nullptr), power(nullptr)
{}

PairDPDDiffCut::~PairDPDDiffCut()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(cut_c);
    memory->destroy(cut_d);
    memory->destroy(power);
  }
}

void PairDPDDiffCut::compute(int eflag, int vflag)
{
  double evdwl = 0.0;
  ev_init(eflag, vflag);

  for (int k = 0; k < 4; ++k) special_sqrt[k] = sqrt(force->special_lj[k]);

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  const int newton_pair = force->newton_pair;
  const double dtinvsqrt = 1.0 / sqrt(update->dt);

  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    const double vxtmp = v[i][0];
    const double vytmp = v[i][1];
    const double vztmp = v[i][2];
    const int itype = type[i];
    int *jlist = list->firstneigh[i];

    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      int j = jlist[jj];
      const int special = sbmask(j);
      const double factor_dpd = special_lj[special];
      const double factor_sqrt = special_sqrt[special];
      j &= NEIGHMASK;

      const double delx = xtmp - x[j][0];
      const double dely = ytmp - x[j][1];
      const double delz = ztmp - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      const int jtype = type[j];

      if (rsq >= cutsq[itype][jtype]) continue;
      const double r = sqrt(rsq);
      if (r < EPSILON) continue;

      const double rinv = 1.0 / r;
      const double delvx = vxtmp - v[j][0];
      const double delvy = vytmp - v[j][1];
      const double delvz = vztmp - v[j][2];
      const double dot = delx * delvx + dely * delvy + delz * delvz;
      const double wc = std::max(1.0 - r / cut_c[itype][jtype], 0.0);
      double wr = std::max(1.0 - r / cut_d[itype][jtype], 0.0);
      wr = pow(wr, 0.5 * power[itype][jtype]);
      const double randnum = random->gaussian();

      double fpair = a0[itype][jtype] * wc;
      fpair -= gamma[itype][jtype] * wr * wr * dot * rinv;
      fpair *= factor_dpd;
      fpair += factor_sqrt * sigma[itype][jtype] * wr * randnum * dtinvsqrt;
      fpair *= rinv;

      f[i][0] += delx * fpair;
      f[i][1] += dely * fpair;
      f[i][2] += delz * fpair;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delx * fpair;
        f[j][1] -= dely * fpair;
        f[j][2] -= delz * fpair;
      }

      if (eflag) {
        evdwl = 0.5 * a0[itype][jtype] * cut_c[itype][jtype] * wc * wc;
        evdwl *= factor_dpd;
      }
      if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

void PairDPDDiffCut::allocate()
{
  PairDPD::allocate();
  const int n = atom->ntypes;
  memory->create(cut_c, n + 1, n + 1, "pair:cut_c");
  memory->create(cut_d, n + 1, n + 1, "pair:cut_d");
  memory->create(power, n + 1, n + 1, "pair:power");
}

void PairDPDDiffCut::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR, "Illegal pair_style dpd/diffcut command");
  temperature = utils::numeric(FLERR, arg[0], false, lmp);
  cut_global = utils::numeric(FLERR, arg[1], false, lmp);
  seed = utils::inumeric(FLERR, arg[2], false, lmp);
  if (seed <= 0) error->all(FLERR, "Pair style dpd/diffcut seed must be positive");

  delete random;
  random = new RanMars(lmp, seed + comm->me);
}

void PairDPDDiffCut::coeff(int narg, char **arg)
{
  if (narg != 7)
    error->all(FLERR, "Incorrect args for pair coefficients: I J A gamma power cutC cutD");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  const double a0_one = utils::numeric(FLERR, arg[2], false, lmp);
  const double gamma_one = utils::numeric(FLERR, arg[3], false, lmp);
  const double power_one = utils::numeric(FLERR, arg[4], false, lmp);
  const double cut_c_one = utils::numeric(FLERR, arg[5], false, lmp);
  const double cut_d_one = utils::numeric(FLERR, arg[6], false, lmp);
  if (power_one <= 0.0 || cut_c_one <= 0.0 || cut_d_one <= 0.0)
    error->all(FLERR, "Pair dpd/diffcut power and cutoffs must be positive");

  int count = 0;
  for (int i = ilo; i <= ihi; ++i)
    for (int j = MAX(jlo, i); j <= jhi; ++j) {
      a0[i][j] = a0_one;
      gamma[i][j] = gamma_one;
      power[i][j] = power_one;
      cut_c[i][j] = cut_c_one;
      cut_d[i][j] = cut_d_one;
      cut[i][j] = MAX(cut_c_one, cut_d_one);
      setflag[i][j] = 1;
      ++count;
    }
  if (count == 0)
    error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
}

double PairDPDDiffCut::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status:\n" + Info::get_pair_coeff_status(lmp));

  sigma[i][j] = sqrt(2.0 * force->boltz * temperature * gamma[i][j]);
  cut[j][i] = cut[i][j];
  cut_c[j][i] = cut_c[i][j];
  cut_d[j][i] = cut_d[i][j];
  power[j][i] = power[i][j];
  a0[j][i] = a0[i][j];
  gamma[j][i] = gamma[i][j];
  sigma[j][i] = sigma[i][j];
  return cut[i][j];
}

void PairDPDDiffCut::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&a0[i][j], sizeof(double), 1, fp);
        fwrite(&gamma[i][j], sizeof(double), 1, fp);
        fwrite(&power[i][j], sizeof(double), 1, fp);
        fwrite(&cut_c[i][j], sizeof(double), 1, fp);
        fwrite(&cut_d[i][j], sizeof(double), 1, fp);
      }
    }
}

void PairDPDDiffCut::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) {
      if (comm->me == 0)
        utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (comm->me == 0) {
          utils::sfread(FLERR, &a0[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &gamma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &power[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut_c[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut_d[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&a0[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&gamma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&power[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut_c[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut_d[i][j], 1, MPI_DOUBLE, 0, world);
        cut[i][j] = MAX(cut_c[i][j], cut_d[i][j]);
      }
    }
}

void PairDPDDiffCut::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; ++i)
    fprintf(fp, "%d %g %g %g %g %g\n", i, a0[i][i], gamma[i][i], power[i][i], cut_c[i][i],
            cut_d[i][i]);
}

void PairDPDDiffCut::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j)
      fprintf(fp, "%d %d %g %g %g %g %g\n", i, j, a0[i][j], gamma[i][j], power[i][j],
              cut_c[i][j], cut_d[i][j]);
}

double PairDPDDiffCut::single(int, int, int itype, int jtype, double rsq, double,
                              double factor_dpd, double &fforce)
{
  const double r = sqrt(rsq);
  if (r < EPSILON || r >= cut_c[itype][jtype]) {
    fforce = 0.0;
    return 0.0;
  }
  const double wc = 1.0 - r / cut_c[itype][jtype];
  fforce = a0[itype][jtype] * wc * factor_dpd / r;
  return factor_dpd * 0.5 * a0[itype][jtype] * cut_c[itype][jtype] * wc * wc;
}
