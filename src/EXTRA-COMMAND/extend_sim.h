#ifdef COMMAND_CLASS
// clang-format off
CommandStyle(extend_sim,ExtendSim);
// clang-format on
#else
#ifndef LMP_EXTEND_SIM_H
#define LMP_EXTEND_SIM_H
#include "command.h"
namespace LAMMPS_NS {
class ExtendSim : public Command {
 public:
  ExtendSim(class LAMMPS *lmp) : Command(lmp) {}
  ~ExtendSim() override;
  void command(int, char **) override;

 private:
  char *restart_file = nullptr;
  char *init_file = nullptr;
  int axis = 2;    // 0=x 1=y 2=z
  double frozen_depth = 0.0;
  double new_end = 0.0;
  int nreplica = 1;

  void parse_args(int, char **);
};
}    // namespace LAMMPS_NS
#endif
#endif
