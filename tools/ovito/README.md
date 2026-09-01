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
* each rank -> transparent `<out>.part.<name>.png` (keep with `--keep-parts`),
  then alpha-composited into `<out>`

Later tasks: deterministic camera module (ov/03), culling (ov/04),
parallel driver (ov/05), on-the-fly polling (ov/06), more views
(ov/07-08), docs (ov/09).
