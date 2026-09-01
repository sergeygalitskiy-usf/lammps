# On-the-fly OVITO rendering for LAMMPS

`render_driver.py` turns the per-MPI-rank dump files of a running LAMMPS
job into composited images, one per view, either after the fact or while
the run is still going.  It is out-of-process: **LAMMPS needs no PYTHON
package**, and a render crash cannot touch the MD.

```
LAMMPS                                   render_driver.py  (separate OVITO process)
  compute PE_All all pe/atom               for each frame:
  dump d1 all custom N dump.%.*  ───────►    read each rank file (that rank's atoms)
    id type x y z c_PE_All                   cull: empty / out-of-view / occluded
  (one file per rank per frame)              render survivors in parallel, shared camera
                                             alpha-composite -> <view>.<step>.png
```

Designed for shock runs that grow long along z (see
`fix wall/piston`, `extend_sim`): the image puts **z along the
horizontal**, x or y along the vertical.

## Requirements

A working OVITO Python.  Any of:

* `/Applications/OVITO.app/Contents/MacOS/ovitos` — the bundled
  interpreter, zero setup; add packages with `ovitos -m pip install ...`
* a **clean** conda env: `conda create -n ovito --strict-channel-priority
  -c https://conda.ovito.org -c conda-forge ovito pillow numpy`
* a plain (non-conda) venv: `python -m venv env && env/bin/pip install
  ovito numpy pillow`

Do **not** mix pip `ovito` into a conda env that also has conda
`pyside6` — the two Qt builds collide
(`NetCDFPluginPython` / `QtConcurrent.framework` import errors).

`numpy` and `pillow` are required; `mpi4py` is **not** (parallelism is
`multiprocessing`).  The driver sets `QT_QPA_PLATFORM=offscreen` and
`OVITO_THREAD_COUNT=1` itself.

## LAMMPS side

```
compute PE_All all pe/atom
dump    d1 all custom 200 dump/cfg.%.*  id type x y z c_PE_All
```

The `%` writes one file per rank (`cfg.<rank>.<step>`); each holds only
that rank's owned atoms, with the **global** box in the header.  The
column order above is what the driver expects; change `--prop` if you
colour by a different last column.

## One frame

```
# one rank file (debugging)
ovitos render_driver.py dump/cfg.0.1000 -o rank0.png

# a whole frame, both ortho views, 4 workers
ovitos render_driver.py 'dump/cfg.*.1000' -o '{view}.{step}.png' \
       --views view1_XZ,view2_YZ --prop c_PE_All --range -9 -6.5 \
       --width 5000 --nproc 4
```

* `--views` — comma list of `view1_XZ`, `view2_YZ`
* `--range LO HI` — colour-coding range for `--prop` (rainbow map)
* `--width` — image width in px (the z axis); height follows the box
  aspect: `H = width * Lx / (z1 - z0)`
* `--zrange Z0 Z1` — fix the horizontal extent; default is `0 .. 1.2*zmax`.
  **Use it for movies** — otherwise a box growing under `extend_sim` /
  shrink-wrap changes the frame height between steps.
* `--radius` — particle sphere radius (box units)
* `--nproc N` — parallel render workers (spawned; output is identical to
  serial)
* `--occlude [--pad D]` — also drop ranks fully covered *and* fully in
  front-shadowed by another rank (optically thick).  Off by default
  because a wrong cull would drop visible atoms.  `--pad` gives the
  footprint test some slack (box units, ~one lattice plane).
* `--keep-parts` — also write each rank's transparent PNG

## On-the-fly (watch mode)

Pass a **directory**; the driver polls it and renders each frame once
all its rank files are present and size-stable:

```
ovitos render_driver.py dump/ -o 'frames/{view}.{step}.png' \
       --views view1_XZ,view2_YZ --range -7 -5 --width 3000 \
       --nproc 4 --until 4000 --idle 30
```

* `{step}` in `-o` is required; `{view}` is required with >1 view
* one worker pool is reused across all frames and views
* the box is read once per frame; each view is culled independently
* stops on `--until <step>`, or after `--idle` seconds with no new frame
* `--nranks N` sets the expected rank-file count (otherwise inferred
  from the first frame); `--interval` sets the poll period

`example/run.sh` runs `mpirun lmp -in in.shock` and the watcher
concurrently, then `ffmpeg`s each view's frames into an mp4.

## Camera

`camera.py` builds the camera as a pure function of the global box plus
a fixed `View` spec — no per-rank input — so every rank of a frame
renders with the identical camera and the parts composite
pixel-for-pixel.  `camera_params()` is OVITO-free and unit tested
(`test_camera.py`); `make_viewport()` is the thin OVITO wrapper.

| view | image right | image up | look dir |
|------|-------------|----------|----------|
| view1_XZ | +z | +x | +y |
| view2_YZ | +z | +y | +x |

## Culling

`cull.py` reads each dump file's header + x,y,z columns (no OVITO) and
returns the ranks worth rendering:

* **empty** — `NUMBER OF ATOMS == 0`
* **frustum** — the rank's atom bounding box, projected onto the image
  plane, misses the view rectangle (catches ranks outside an explicit
  `--zrange`, or above `1.2*zmax`)
* **occlusion** (`--occlude`) — another rank shares this one's
  horizontal+vertical footprint and lies entirely in front along the
  depth axis; exact for an axis-aligned processor grid viewed along a
  box axis, conservative otherwise

`empty` and `frustum` are always on and never drop a visible atom.

## Tests

```
python3 test_camera.py     # pure math, no OVITO
python3 test_cull.py       # synthetic dump files, no OVITO
```

## Not done yet

* tilted 3-D views (view3 / view4) and depth-ordered "over" compositing
  for ranks that overlap in screen space — OVITO Python exposes no
  per-pixel depth buffer, so this needs per-rank depth ordering
