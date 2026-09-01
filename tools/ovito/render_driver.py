#!/usr/bin/env ovitos
"""
render_driver.py  --  out-of-process OVITO renderer for LAMMPS per-rank dumps.

LAMMPS writes one dump file per MPI rank per frame:

    compute PE_All all pe/atom
    dump    d1 all custom 100 dump.%.*  id type x y z c_PE_All

This driver renders each surviving rank file with an identical camera
and composites the transparent parts into one image per view.

ov/01  single dump file -> one XZ image
ov/02  all rank files of a frame -> one composited image
ov/03  deterministic camera module (camera.py)
ov/04  culling: empty + frustum/z-clip (always), occlusion (--occlude)
ov/05  parallel render (multiprocessing, spawn) + in-memory merge

Run with a working OVITO Python:

    ovitos render_driver.py 'dump.*.1000' -o view1_XZ.1000.png --nproc 8
    ovitos render_driver.py dump.0.1000   -o rank0.png                 # single file
"""

import argparse
import glob
import multiprocessing
import os
import re
import tempfile
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OVITO_THREAD_COUNT", "1")

import numpy as np
from ovito.io import import_file
from ovito.modifiers import ColorCodingModifier
from ovito.vis import TachyonRenderer
from PIL import Image

from camera import VIEWS, camera_params, make_viewport
from cull import visible

# LAMMPS 'dump custom id type x y z c_PE_All' -> OVITO property names
DUMP_COLUMNS = [
    "Particle Identifier",
    "Particle Type",
    "Position.X",
    "Position.Y",
    "Position.Z",
    "c_PE_All",
]


def _render_worker(task):
    """Render one rank file -> (H, W, 4) uint8 RGBA array.  Picklable / spawn-safe."""
    dump_path, params, prop, crange, radius = task
    pipeline = import_file(dump_path, columns=DUMP_COLUMNS)
    pipeline.source.data.particles_.vis.radius = radius
    pipeline.source.data.cell_.vis.enabled = False
    pipeline.modifiers.append(ColorCodingModifier(
        property=prop, start_value=crange[0], end_value=crange[1],
        gradient=ColorCodingModifier.Rainbow()))
    pipeline.add_to_scene()
    vp = make_viewport(params)
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        vp.render_image(size=params["size"], filename=tmp, alpha=True,
                        background=(0, 0, 0), renderer=TachyonRenderer())
        arr = np.asarray(Image.open(tmp).convert("RGBA"))
    finally:
        os.unlink(tmp)
        pipeline.remove_from_scene()
    return arr


def render_frame(rank_files, out_path, view="view1_XZ", prop="c_PE_All",
                 crange=(-9.0, -6.5), width=5000, zrange=None, radius=0.5,
                 keep_parts=False, occlude=False, pad=0.0, nproc=1, pool=None):
    rank_files = sorted(rank_files)
    if not rank_files:
        raise SystemExit("no dump files matched")

    v = VIEWS[view]
    if width:
        v = replace(v, width_px=width)
    # camera from the first file's global box (identical in every rank file)
    box = import_file(rank_files[0], columns=DUMP_COLUMNS).compute().cell[...]
    params = camera_params(v, box, horiz_range=zrange)
    W, H = params["size"]

    keep, stats = visible(rank_files, v, params, occlude=occlude, pad=pad)
    print(f"  culled: {stats['empty']} empty, {stats['frustum']} out-of-view, "
          f"{stats['occluded']} occluded -> {stats['kept']}/{len(rank_files)} rendered")
    if not keep:
        Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(out_path)
        return W, H, 0

    tasks = [(f, params, prop, tuple(crange), radius) for f in keep]
    if pool is not None:
        arrays = pool.map(_render_worker, tasks)
    elif max(1, min(nproc, len(tasks))) == 1:
        arrays = [_render_worker(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(min(nproc, len(tasks))) as p:
            arrays = p.map(_render_worker, tasks)

    merged = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for f, arr in zip(keep, arrays):
        part = Image.fromarray(arr, "RGBA")
        if keep_parts:
            part.save(f"{out_path}.part.{os.path.basename(f)}.png")
        merged = Image.alpha_composite(merged, part)
    merged.save(out_path)
    return W, H, len(arrays)


_FRAME_RE = re.compile(r"^(?P<prefix>.+)\.(?P<rank>\d+)\.(?P<step>\d+)$")


def scan_frames(dumpdir):
    """{step: [rank_files]} for files named  <prefix>.<rank>.<step>  in dumpdir."""
    frames = {}
    for name in os.listdir(dumpdir):
        m = _FRAME_RE.match(name)
        if m:
            frames.setdefault(int(m["step"]), []).append(os.path.join(dumpdir, name))
    return frames


def _stable(files, sizes):
    """True if every file's size is unchanged since the sizes dict was last filled."""
    ok = True
    for f in files:
        try:
            s = os.path.getsize(f)
        except OSError:
            return False
        ok = ok and sizes.get(f) == s
        sizes[f] = s
    return ok


def watch(dumpdir, out_tmpl, nranks=None, interval=2.0, idle=60.0, until=None,
          nproc=1, **frame_kw):
    """Render each new complete+stable frame as it appears in dumpdir."""
    if "{step}" not in out_tmpl:
        raise SystemExit("watch mode needs '{step}' in -o, e.g. -o 'frames/v1.{step}.png'")
    os.makedirs(os.path.dirname(out_tmpl) or ".", exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(nproc) if nproc > 1 else None
    done, sizes, last_new = set(), {}, time.time()
    print(f"watching {dumpdir}  (interval {interval}s, idle stop {idle}s)")
    try:
        while True:
            frames = scan_frames(dumpdir)
            if nranks is None and frames:
                nranks = max(len(v) for v in frames.values())
                print(f"  inferred nranks = {nranks}")
            progressed = False
            for step in sorted(frames):
                if step in done or (until is not None and step > until):
                    continue
                files = frames[step]
                if nranks and len(files) < nranks:
                    continue
                if not _stable(files, sizes):
                    continue
                out = out_tmpl.format(step=step)
                t0 = time.time()
                print(f"[step {step}] {len(files)} rank files -> {out}")
                w, h, n = render_frame(files, out, nproc=nproc, pool=pool, **frame_kw)
                print(f"  {w}x{h}  {n} rendered  {time.time()-t0:.1f}s")
                done.add(step)
                progressed = True
            now = time.time()
            if progressed:
                last_new = now
            elif now - last_new > idle:
                print("idle timeout, stopping")
                break
            if until is not None and done and max(done) >= until:
                print(f"reached step {until}, stopping")
                break
            time.sleep(interval)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return sorted(done)


def main():
    ap = argparse.ArgumentParser(description="render LAMMPS per-rank dump(s) to an XZ image")
    ap.add_argument("dumps", help="a dump file / glob for one frame, OR a directory to watch "
                                  "(quote globs, e.g. 'dump.*.1000')")
    ap.add_argument("-o", "--out", default="view1_XZ.png")
    ap.add_argument("--view", default="view1_XZ", choices=sorted(VIEWS))
    ap.add_argument("--width", type=int, default=5000, help="image width in px (z horizontal)")
    ap.add_argument("--prop", default="c_PE_All")
    ap.add_argument("--range", nargs=2, type=float, default=[-9.0, -6.5], metavar=("LO", "HI"))
    ap.add_argument("--zrange", nargs=2, type=float, default=None, metavar=("Z0", "Z1"),
                    help="fixed horizontal z-range; default 0 .. 1.2*zmax")
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel render workers (spawned)")
    ap.add_argument("--keep-parts", action="store_true", help="also write per-rank PNGs")
    ap.add_argument("--occlude", action="store_true",
                    help="also drop ranks fully hidden behind another (optically thick)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="slack (box units) for the occlusion footprint test")
    ap.add_argument("--nranks", type=int, default=None,
                    help="watch mode: expected rank-file count per frame (else inferred)")
    ap.add_argument("--interval", type=float, default=2.0, help="watch mode: poll seconds")
    ap.add_argument("--idle", type=float, default=60.0,
                    help="watch mode: stop after this many seconds with no new frame")
    ap.add_argument("--until", type=int, default=None, help="watch mode: stop after this step")
    a = ap.parse_args()

    frame_kw = dict(view=a.view, prop=a.prop, crange=tuple(a.range), width=a.width,
                    zrange=tuple(a.zrange) if a.zrange else None, radius=a.radius,
                    keep_parts=a.keep_parts, occlude=a.occlude, pad=a.pad)

    if os.path.isdir(a.dumps):
        steps = watch(a.dumps, a.out, nranks=a.nranks, interval=a.interval, idle=a.idle,
                      until=a.until, nproc=a.nproc, **frame_kw)
        print(f"done: {len(steps)} frames")
    else:
        files = glob.glob(a.dumps) if any(c in a.dumps for c in "*?[") else [a.dumps]
        w, h, n = render_frame(files, a.out, nproc=a.nproc, **frame_kw)
        print(f"wrote {a.out}  {w}x{h} px  from {n} rank file(s)")


if __name__ == "__main__":
    main()
