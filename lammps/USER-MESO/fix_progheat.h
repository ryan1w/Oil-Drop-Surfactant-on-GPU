#ifdef FIX_CLASS

FixStyle(progheat,FixProgHeat)

#else

#ifndef LMP_FIX_PROGHEAT
#define LMP_FIX_PROGHEAT

#include "fix.h"
#include <string>

namespace LAMMPS_NS {

class FixProgHeat : public Fix {
public:
    FixProgHeat(class LAMMPS *, int, char **);
    ~FixProgHeat();
    virtual int setmask();
    virtual void setup(int);
    virtual void post_force(int);
    virtual double compute_scalar();

private:
    double delta, Cv, dQ;
    std::string T_template; // T_expr = f(t)

    char **T_command;
    std::string t_user, T_varname;
};

}

#endif

#endif
