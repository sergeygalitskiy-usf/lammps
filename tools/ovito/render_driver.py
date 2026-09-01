#!/usr/bin/env ovitos
"""
render_driver.py  --  out-of-process OVITO renderer for LAMMPS per-rank dumps.

LAMMPS writes one dump file per MPI rank per frame:

    compute PE_All all pe/atom
    dump    d1 all custom 100 dump.%.*  id type x y z c_PE_All

This driver renders each rank file with an identical camera and
composites the transparent-background parts into one image per view.

ov/01: single dump file -> one XZ image.
ov/02: all rank files of a frame -> one composited XZ image (file-based
       merge, no culling yet).

Run with a working OVITO Python:

    ovitos render_driver.py 'dump.*.1000' -o view1_XZ.1000.png
    ovitos render_driver.py dump.0.1000   -o rank0.png            # single file
"""

import argparse
import glob
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OVITO_THREAD_COUNT", "1")

from ovito.io import import_file
from ovito.modifiers import ColorCodingModifier
from ovito.vis import TachyonRenderer, Viewport
from PIL import Image

# LAMMPS 'dump custom id type x y z c_PE_All' -> OVITO property names
DUMP_COLUMNS = [
    "Particle Identifier",
    "Particle Type",
    "Position.X",
    "Position.Y",
    "Position.Z",
    "c_PE_All",
]


def xz_camera(box_matrix, width, zrange):
    """Deterministic ortho camera for an XZ view, from the *global* box.

    Returns (viewport, (W, H)).  Image horizontal = +z, vertical = +x.
    Every rank file of a frame carries the same global box, so every
    rank renders with the same camera -> parts are pixel-aligned.
    """
    origin = box_matrix[:, 3]
    Lx = box_matrix[0, 0]
    Lz = box_matrix[2, 2]
    zmax = origin[2] + Lz
    z0, z1 = zrange if zrange else (0.0, 1.2 * zmax)

    W = int(width)
    H = max(64, int(round(W * Lx / (z1 - z0))))

    vp = Viewport(type=Viewport.Type.Ortho)
    vp.camera_dir = (0.0, 1.0, 0.0)          # look along +y  -> right = +z, up = +x
    vp.camera_up = (1.0, 0.0, 0.0)
    vp.camera_pos = (origin[0] + Lx / 2.0,
                     origin[1] - 10.0 * max(Lx, Lz),
                     0.5 * (z0 + z1))
    vp.fov = Lx / 2.0                        # ortho half-height (x extent), box units
    return vp, (W, H)


def render_rank(dump_path, out_png, size, vp, prop, crange, radius):
    """Render one rank file to a transparent-background PNG."""
    pipeline = import_file(dump_path, columns=DUMP_COLUMNS)
    pipeline.source.data.particles_.vis.radius = radius
    pipeline.source.data.cell_.vis.enabled = False
    pipeline.modifiers.append(ColorCodingModifier(
        property=prop, start_value=crange[0], end_value=crange[1],
        gradient=ColorCodingModifier.Rainbow()))
    pipeline.add_to_scene()
    vp.render_image(size=size, filename=out_png, alpha=True,
                    background=(0, 0, 0), renderer=TachyonRenderer())
    pipeline.remove_from_scene()


def render_frame(rank_files, out_path, prop="c_PE_All", crange=(-9.0, -6.5),
                 width=5000, zrange=None, radius=0.5, keep_parts=False):
    rank_files = sorted(rank_files)
    if not rank_files:
        raise SystemExit("no dump files matched")

    # camera from the first file's global box (identical in every rank file)
    box = import_file(rank_files[0], columns=DUMP_COLUMNS).compute().cell[...]
    vp, size = xz_camera(box, width, zrange)
    W, H = size

    parts = []
    for f in rank_files:
        p = f"{out_path}.part.{os.path.basename(f)}.png"
        render_rank(f, p, size, vp, prop, crange, radius)
        parts.append(p)

    merged = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for p in parts:
        merged = Image.alpha_composite(merged, Image.open(p).convert("RGBA"))
    merged.save(out_path)

    if not keep_parts:
        for p in parts:
            os.remove(p)
    return W, H, len(rank_files)


def main():
    ap = argparse.ArgumentParser(description="render LAMMPS per-rank dump(s) to an XZ image")
    ap.add_argument("dumps", help="one dump file, or a glob for a frame's rank files "
                                  "(quote it, e.g. 'dump.*.1000')")
    ap.add_argument("-o", "--out", default="view1_XZ.png")
    ap.add_argument("--width", type=int, default=5000, help="image width in px (z horizontal)")
    ap.add_argument("--prop", default="c_PE_All")
    ap.add_argument("--range", nargs=2, type=float, default=[-9.0, -6.5], metavar=("LO", "HI"))
    ap.add_argument("--zrange", nargs=2, type=float, default=None, metavar=("Z0", "Z1"),
                    help="fixed horizontal z-range; default 0 .. 1.2*zmax")
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--keep-parts", action="store_true", help="do not delete per-rank PNGs")
    a = ap.parse_args()

    files = glob.glob(a.dumps) if any(c in a.dumps for c in "*?[") else [a.dumps]
    w, h, n = render_frame(files, a.out, a.prop, tuple(a.range), a.width,
                           tuple(a.zrange) if a.zrange else None, a.radius, a.keep_parts)
    print(f"wrote {a.out}  {w}x{h} px  from {n} rank file(s)")


if __name__ == "__main__":
    main()
