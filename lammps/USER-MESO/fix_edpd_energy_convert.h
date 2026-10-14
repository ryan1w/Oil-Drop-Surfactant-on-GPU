#ifdef FIX_CLASS

FixStyle(ecvt/edpd,FixEDPDEnergyConvert)

#else

#ifndef LMP_FIX_EDPD_ENERGY_CONVERT
#define LMP_FIX_EDPD_ENERGY_CONVERT

#include "fix.h"

namespace LAMMPS_NS {

class FixEDPDEnergyConvert : public Fix
{
public:
    FixEDPDEnergyConvert(LAMMPS *lmp, int narg, char **arg);
    ~FixEDPDEnergyConvert(){}
    virtual int setmask();
    virtual void initial_integrate();
    void init();
    void init_list(int, class NeighList *);
    virtual int pack_comm(int, int *, double *, int, int *);
    virtual void unpack_comm(int, int, double *);

protected:
    int jgroup, jgroupbit;
    double **cutsq;
    class Pair *pair;
    class NeighList *list;
    double theta_one, da_one, omega_one;
};

}

#endif

#endif
