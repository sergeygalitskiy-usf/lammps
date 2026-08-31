#include "extend_sim.h"

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "utils.h"

#include "fmt/format.h"

#include <cstdio>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ExtendSim::~ExtendSim()
{
  delete[] restart_file;
  delete[] init_file;
}

/* ----------------------------------------------------------------------
   extend_sim init <file> nreplica <N>
              [restart <file>] [axis <x|y|z>] [new_end <coord>]
              [gap <d>] [frozen_depth <d>]

   With 'restart', that restart file is loaded as the base system (no box
   may exist yet). Without it, the command operates on the already-loaded
   system. It then appends N copies of the <init> sample onto the end
   along <axis>. N <= 0 is a no-op.

   A periodic/fixed +axis face is grown with change_box before atoms are
   added; a shrink-wrapped face is left to auto-expand inside read_data.
   'gap' inserts empty space below the first appended slab -- use it when
   stacking onto a non-periodic surface so the new slab does not overlap
   the existing surface layer (default 0 = seamless periodic tiling).
   'frozen_depth' is optional and currently only range-checked against the
   sample thickness; freezing itself is done in the input script.
------------------------------------------------------------------------- */

void ExtendSim::command(int narg, char **arg)
{
  parse_args(narg, arg);

  const char axischar = "xyz"[axis];

  // N <= 0 -> nothing to do
  if (nreplica <= 0) {
    if (comm->me == 0) utils::logmesg(lmp, "extend_sim: nreplica = {}, nothing to do\n", nreplica);
    return;
  }

  // 1. obtain the base system
  if (restart_file) {
    if (domain->box_exist)
      error->all(FLERR,
                 "extend_sim: 'restart' was given but a simulation box already exists; "
                 "run 'clear' first or omit the restart keyword");
    input->one(fmt::format("read_restart {}", restart_file));
    if (!domain->box_exist)
      error->all(FLERR, "extend_sim: reading restart '{}' did not create a box", restart_file);
  } else if (!domain->box_exist) {
    error->all(FLERR,
               "extend_sim: no simulation box to extend; load a system first or pass "
               "'restart <file>'");
  }

  const double lo = domain->boxlo[axis];
  const double hi_old = domain->boxhi[axis];

  // 2. measure the pristine sample thickness from the init data file header
  double dlo, dhi;
  data_file_extent(init_file, axis, dlo, dhi);
  const double zl = dhi - dlo;
  if (zl <= 0.0)
    error->all(FLERR, "extend_sim: non-positive sample thickness ({:.6g}) from '{}'", zl,
               init_file);

  // +axis hi boundary: 0=periodic, 1=fixed, 2/3=shrink-wrapped.
  // Periodic/fixed boxes must be grown before atoms are added; a
  // shrink-wrapped box re-expands on its own inside read_data add.
  const int bhi = domain->boundary[axis][1];
  const bool grow_box = (bhi == 0 || bhi == 1);

  // 3. resolve the new box end: default is gap + N stacked slabs
  const double stacked_end = hi_old + gap + nreplica * zl;
  if (!new_end_set) new_end = stacked_end;
  if (grow_box && new_end < stacked_end - 1.0e-6)
    error->all(FLERR,
               "extend_sim: new_end ({:.6g}) is below the stacked height ({:.6g}) for "
               "{} replicas of thickness {:.6g}",
               new_end, stacked_end, nreplica, zl);
  if (frozen_depth > 0.0 && frozen_depth >= zl)
    error->all(FLERR,
               "extend_sim: frozen_depth ({:.6g}) must be smaller than the sample "
               "thickness ({:.6g})",
               frozen_depth, zl);

  if (comm->me == 0) {
    if (grow_box)
      utils::logmesg(lmp,
                     "extend_sim: sample thickness {:.6g} along {}; stacking {} replica(s) of "
                     "'{}'; box hi {:.6g} -> {:.6g}\n",
                     zl, axischar, nreplica, init_file, hi_old, new_end);
    else
      utils::logmesg(lmp,
                     "extend_sim: sample thickness {:.6g} along {}; stacking {} replica(s) of "
                     "'{}'; {} hi boundary shrink-wrapped, box will auto-expand\n",
                     zl, axischar, nreplica, init_file, axischar);
  }

  // 4. grow the simulation box (fixed / periodic +axis face only)
  if (grow_box)
    input->one(
        fmt::format("change_box all {} final {:.16g} {:.16g} units box", axischar, lo, new_end));

  // 5. append shifted copies: replica i bottom sits at hi_old + i*zl,
  //    correcting for the sample's own lower bound dlo
  for (int i = 0; i < nreplica; ++i) {
    double s[3] = {0.0, 0.0, 0.0};
    s[axis] = (hi_old - dlo) + gap + i * zl;
    input->one(fmt::format("read_data {} add append shift {:.16g} {:.16g} {:.16g}", init_file, s[0],
                           s[1], s[2]));
  }

  // freezing (region / group / fix setforce) is left to the input script
}

/* ----------------------------------------------------------------------
   read the [dim]lo / [dim]hi box bounds from a LAMMPS data file header
------------------------------------------------------------------------- */

void ExtendSim::data_file_extent(const char *file, int dim, double &lo, double &hi)
{
  const char *tag = (dim == 0) ? "xlo xhi" : (dim == 1) ? "ylo yhi" : "zlo zhi";
  lo = hi = 0.0;
  int found = 0;

  if (comm->me == 0) {
    FILE *fp = fopen(file, "r");
    if (!fp) error->one(FLERR, "extend_sim: cannot open init data file '{}'", file);
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
      if (strstr(line, tag)) {
        if (sscanf(line, "%lg %lg", &lo, &hi) == 2) found = 1;
        break;
      }
      // header ends once a body section begins
      if (strstr(line, "Atoms") || strstr(line, "Masses") || strstr(line, "Velocities")) break;
    }
    fclose(fp);
  }

  MPI_Bcast(&found, 1, MPI_INT, 0, world);
  MPI_Bcast(&lo, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&hi, 1, MPI_DOUBLE, 0, world);
  if (!found) error->all(FLERR, "extend_sim: could not find '{}' bounds in '{}'", tag, file);
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
      new_end_set = true;
      iarg += 2;

    } else if (strcmp(arg[iarg], "gap") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "extend_sim gap", error);
      gap = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
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
  if (!init_file) error->all(FLERR, "extend_sim: 'init <file>' is required");
  if (nreplica < 0) error->all(FLERR, "extend_sim: nreplica cannot be negative");
  if (gap < 0.0) error->all(FLERR, "extend_sim: gap cannot be negative");
  if (frozen_depth < 0.0) error->all(FLERR, "extend_sim: frozen_depth cannot be negative");

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "extend_sim args: base={} init={} axis={} frozen_depth={:.6g} "
                   "new_end={:.6g} nreplica={}\n",
                   restart_file ? restart_file : "(current system)", init_file, "xyz"[axis],
                   frozen_depth, new_end, nreplica);
}
