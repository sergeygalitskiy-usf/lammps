#include "extend_sim.h"

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "utils.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ExtendSim::~ExtendSim()
{
  delete[] restart_file;
  delete[] init_file;
}

/* ----------------------------------------------------------------------
   extend_sim restart <file> init <file> axis <x|y|z>
              frozen_depth <d> new_end <coord> nreplica <N>
------------------------------------------------------------------------- */

void ExtendSim::command(int narg, char **arg)
{
  if (domain->box_exist)
    error->all(FLERR, "extend_sim must be run before the simulation box exists");

  parse_args(narg, arg);

  // box extension + replica insertion + frozen groups land in es/04 / es/05
  error->all(FLERR, "extend_sim: argument parsing complete; core logic not implemented yet");
}

/* ---------------------------------------------------------------------- */

void ExtendSim::parse_args(int narg, char **arg)
{
  if (narg < 1) utils::missing_cmd_args(FLERR, "extend_sim", error);

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "restart") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "extend_sim restart", error);
      delete[] restart_file;
      restart_file = utils::strdup(arg[iarg + 1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "init") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "extend_sim init", error);
      delete[] init_file;
      init_file = utils::strdup(arg[iarg + 1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "axis") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "extend_sim axis", error);
      if (strcmp(arg[iarg + 1], "x") == 0)
        axis = 0;
      else if (strcmp(arg[iarg + 1], "y") == 0)
        axis = 1;
      else if (strcmp(arg[iarg + 1], "z") == 0)
        axis = 2;
      else
        error->all(FLERR, "extend_sim: axis must be x, y, or z");
      iarg += 2;

    } else if (strcmp(arg[iarg], "frozen_depth") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "extend_sim frozen_depth", error);
      frozen_depth = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg], "new_end") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "extend_sim new_end", error);
      new_end = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg], "nreplica") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "extend_sim nreplica", error);
      nreplica = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;

    } else {
      error->all(FLERR, "Unknown extend_sim keyword: {}", arg[iarg]);
    }
  }

  // ---- semantic validation ----
  if (!restart_file) error->all(FLERR, "extend_sim: 'restart <file>' is required");
  if (!init_file) error->all(FLERR, "extend_sim: 'init <file>' is required");
  if (nreplica < 1) error->all(FLERR, "extend_sim: nreplica must be >= 1");
  if (frozen_depth <= 0.0) error->all(FLERR, "extend_sim: frozen_depth must be > 0");

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "extend_sim args: restart={} init={} axis={} frozen_depth={:.6g} "
                   "new_end={:.6g} nreplica={}\n",
                   restart_file, init_file, "xyz"[axis], frozen_depth, new_end, nreplica);
}
