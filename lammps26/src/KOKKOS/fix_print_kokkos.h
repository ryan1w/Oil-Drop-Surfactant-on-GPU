/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(print/kk,FixPrintKokkos<LMPDeviceType>);
FixStyle(print/kk/device,FixPrintKokkos<LMPDeviceType>);
FixStyle(print/kk/host,FixPrintKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_PRINT_KOKKOS_H
#define LMP_FIX_PRINT_KOKKOS_H

#include "fix_print.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

// fix print touches no per-atom data of its own: it only substitutes
// variables, and the variables bring to the host whatever per-atom data they
// read (VariableKokkos syncs atom vectors, group and special functions, fix
// references and non-KOKKOS computes; KOKKOS computes on the device sync
// their own input).  The non-KOKKOS style still has the default datamasks
// (ALL_MASK), so ModifyKokkos copies every atom array to the host and marks
// it host-modified around each call: with "fix print 1 ..." that is a full
// host<->device round trip every step.  This subclass only declares the
// empty datamasks.  It is not a "fake port": there is no per-atom work that
// could move to the device.

template<class DeviceType>
class FixPrintKokkos : public FixPrint {
 public:
  FixPrintKokkos(class LAMMPS *, int, char **);
};

}    // namespace LAMMPS_NS

#endif
#endif
