/* ----------------------------------------------------------------------
   Device implementation of compute com using GroupKokkos reductions.
------------------------------------------------------------------------- */

#include "compute_com_kokkos.h"

#include "group_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

template<class DeviceType>
ComputeCOMKokkos<DeviceType>::ComputeCOMKokkos(LAMMPS *lmp, int narg, char **arg) :
    ComputeCOM(lmp, narg, arg), masstotal_kk(0.0)
{
  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
}

template<class DeviceType>
void ComputeCOMKokkos<DeviceType>::init()
{
  auto *groupKK = static_cast<GroupKokkos *>(group);
  masstotal_kk = groupKK->template mass_kk<DeviceType>(igroup);
}

template<class DeviceType>
void ComputeCOMKokkos<DeviceType>::compute_vector()
{
  invoked_vector = update->ntimestep;
  auto *groupKK = static_cast<GroupKokkos *>(group);
  if (group->dynamic[igroup]) masstotal_kk = groupKK->template mass_kk<DeviceType>(igroup);
  groupKK->template xcm_kk<DeviceType>(igroup, masstotal_kk, vector);
}

namespace LAMMPS_NS {
template class ComputeCOMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeCOMKokkos<LMPHostType>;
#endif
}
