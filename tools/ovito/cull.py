"""Cull LAMMPS per-rank dump files that add nothing to a view.

Three tests, cheapest first:

  empty      NUMBER OF ATOMS == 0
  frustum    the rank's atom AABB, projected onto the image plane, does
             not overlap the view rectangle (catches ranks entirely
             outside an explicit zrange, or above 1.2*zmax)
  occlusion  (opt-in) another rank with the same horizontal+vertical
             footprint lies entirely in front along the depth axis, and
             the system is optically thick -> this rank is hidden

Only `empty` and `frustum` run by default; they can never drop a
visible atom.  `occlusion` is exact for an axis-aligned processor grid
viewed along a box axis and conservative otherwise (it only culls a
rank fully covered *and* fully occluded by a single other rank).

Bounds come from the dump header + a cheap read of the x,y,z columns
(no OVITO needed), so this module is pure and unit-testable.
"""

import numpy as np

# 'ITEM: TIMESTEP'(2) 'NUMBER OF ATOMS'(2) 'BOX BOUNDS'(4) 'ATOMS ...'(1)
_HEADER_LINES = 9


def read_bounds(path):
    """-> dict(natoms, aabb=[[lo,hi]*3], box=3x4).  aabb is None if empty."""
    with open(path) as fh:
        lines = [next(fh) for _ in range(_HEADER_LINES)]
    natoms = int(lines[3])
    box = np.zeros((3, 4))
    for i in range(3):
        lo, hi = (float(v) for v in lines[5 + i].split()[:2])
        box[i, i] = hi - lo
        box[i, 3] = lo

    aabb = None
    if natoms:
        xyz = np.loadtxt(path, skiprows=_HEADER_LINES, usecols=(2, 3, 4),
                         ndmin=2)
        aabb = np.stack([xyz.min(axis=0), xyz.max(axis=0)], axis=1)  # (3,2)
    return {"path": path, "natoms": natoms, "aabb": aabb, "box": box}


def _overlap(lo_a, hi_a, lo_b, hi_b):
    return not (hi_a < lo_b or lo_a > hi_b)


def _covers(s, r, axis, pad):
    return s[axis][0] <= r[axis][0] + pad and s[axis][1] >= r[axis][1] - pad


def visible(files, view, params, occlude=False, pad=0.0):
    """Return (keep_paths, stats) where stats = dict(empty, frustum, occluded, kept)."""
    ha, va, da = view.horiz_axis, view.vert_axis, view.depth_axis
    vh, vv = params["horiz"], params["vert"]

    recs, empty = [], 0
    for f in files:
        b = read_bounds(f)
        if b["natoms"] == 0:
            empty += 1
            continue
        recs.append(b)

    # frustum / clip
    in_view, frustum = [], 0
    for b in recs:
        a = b["aabb"]
        if _overlap(a[ha][0], a[ha][1], vh[0], vh[1]) and \
           _overlap(a[va][0], a[va][1], vv[0], vv[1]):
            in_view.append(b)
        else:
            frustum += 1

    # occlusion
    keep, occluded = in_view, 0
    if occlude and len(in_view) > 1:
        keep = []
        for r in in_view:
            ra = r["aabb"]
            hidden = any(
                s is not r
                and _covers(s["aabb"], ra, ha, pad)
                and _covers(s["aabb"], ra, va, pad)
                and s["aabb"][da][1] <= ra[da][0] + pad      # s fully in front
                for s in in_view
            )
            if hidden:
                occluded += 1
            else:
                keep.append(r)

    stats = {"empty": empty, "frustum": frustum, "occluded": occluded,
             "kept": len(keep)}
    return [b["path"] for b in keep], stats
