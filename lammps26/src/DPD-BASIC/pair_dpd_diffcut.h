/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(dpd/diffcut,PairDPDDiffCut);
// clang-format on
#else

#ifndef LMP_PAIR_DPD_DIFFCUT_H
#define LMP_PAIR_DPD_DIFFCUT_H

#include "pair_dpd.h"

namespace LAMMPS_NS {

class PairDPDDiffCut : public PairDPD {
 public:
  PairDPDDiffCut(class LAMMPS *);
  ~PairDPDDiffCut() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  double **cut_c, **cut_d, **power;

  void allocate() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
