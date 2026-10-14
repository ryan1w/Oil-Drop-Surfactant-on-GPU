/* -*- c++ -*- ---------------------------------------------------------- */
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(com/kk,ComputeCOMKokkos<LMPDeviceType>);
ComputeStyle(com/kk/device,ComputeCOMKokkos<LMPDeviceType>);
ComputeStyle(com/kk/host,ComputeCOMKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_COM_KOKKOS_H
#define LMP_COMPUTE_COM_KOKKOS_H

#include "compute_com.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class ComputeCOMKokkos : public ComputeCOM {
 public:
  ComputeCOMKokkos(class LAMMPS *, int, char **);
  void init() override;
  void compute_vector() override;

 private:
  double masstotal_kk;
};

}    // namespace LAMMPS_NS
#endif
#endif
