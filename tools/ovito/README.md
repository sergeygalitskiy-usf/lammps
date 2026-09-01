# On-the-fly OVITO rendering for LAMMPS

Out-of-process renderer: LAMMPS writes **per-rank** dump files, a separate
OVITO process reads them, culls, renders each, and composites into one
image per view.  No LAMMPS PYTHON package is required.

## OVITO interpreter

Any working OVITO Python works.  Two known-good options on this machine:

* `/Applications/OVITO.app/Contents/MacOS/ovitos`  (bundled, zero setup)
* `~/miniconda3/envs/ovito/bin/python`  (conda env from https://conda.ovito.org;
  `pillow` added via conda-forge)

Do **not** mix pip `ovito` into a conda env with conda `pyside6` -- the Qt
libraries collide (`NetCDFPluginPython` / `QtConcurrent.framework` errors).

Headless: the driver sets `QT_QPA_PLATFORM=offscreen` and
`OVITO_THREAD_COUNT=1` itself.

## LAMMPS side

```
compute PE_All all pe/atom
dump    d1 all custom 100 dump.%.*  id type x y z c_PE_All
```

The `%` makes one file per MPI rank (`dump.<rank>.<step>`); each holds
only that rank's owned atoms, with the global box in the header.

## render_driver.py

Render one dump file, or a whole frame's rank files, to one XZ image.

```
# one rank file (debugging)
ovitos render_driver.py dump.0.100 -o rank0.png

# a full frame: all rank files -> identical camera -> composited
ovitos render_driver.py 'dump.*.1000' -o view1_XZ.1000.png \
       --width 5000 --prop c_PE_All --range -9 -6.5
```

* image horizontal = +z (shock axis), vertical = +x
* height = width * Lx / (z1 - z0)
* z-range defaults to 0 .. 1.2*zmax; override with `--zrange Z0 Z1`
  (needed for a movie -- otherwise a growing box changes the frame height)
* `--nproc N` renders the surviving ranks through a spawned pool
* `--occlude [--pad D]` also drops ranks hidden behind another

## on-the-fly (watch mode)

Point the first argument at a **directory**; the driver polls it and
renders each frame as soon as all its rank files are present and stable:

```
ovitos render_driver.py dump/ -o 'frames/view1_XZ.{step}.png' \
       --range -7 -5 --width 3000 --nproc 4 --until 4000 --idle 30
```

`{step}` in `-o` is required.  One worker pool is reused across frames.
Stops on `--until <step>` or after `--idle` seconds with no new frame.

`example/run.sh` runs `mpirun lmp -in in.shock` and the watcher together,
then `ffmpeg`s `frames/*.png` into `view1_XZ.mp4`.

Remaining: View abstraction + view2_YZ (ov/07), tilted 3-D views +
depth compositing (ov/08), docs (ov/09).
