#!/usr/bin/env ovitos
"""
render_driver.py  --  out-of-process OVITO renderer for LAMMPS per-rank dumps.

ov/01: minimal version -- render ONE LAMMPS dump file to one XZ image,
colour-coded by a per-atom property (default c_PE_All) with a rainbow
map over a fixed range.  Later tasks add multi-rank merge, culling, a
deterministic camera module, more views, and on-the-fly polling.

Run with the OVITO app's bundled interpreter:

    /Applications/OVITO.app/Contents/MacOS/ovitos render_driver.py dump.0.100 -o view1_XZ.png
"""

import argparse
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OVITO_THREAD_COUNT", "1")

import numpy as np
from ovito.io import import_file
from ovito.modifiers import ColorCodingModifier
from ovito.vis import TachyonRenderer, Viewport

# LAMMPS 'dump custom id type x y z c_PE_All' -> OVITO property names
DUMP_COLUMNS = [
    "Particle Identifier",
    "Particle Type",
    "Position.X",
    "Position.Y",
    "Position.Z",
    "c_PE_All",
]


def render_xz(dump_path, out_path, prop="c_PE_All", crange=(-9.0, -6.5),
              width=5000, zrange=None, radius=0.5):
    pipeline = import_file(dump_path, columns=DUMP_COLUMNS)
    data = pipeline.compute()
    n = data.particles.count

    # global box from the dump header
    m = data.cell[...]                      # 3x4 : columns are the 3 edge vectors + origin
    origin = m[:, 3]
    Lx, Ly, Lz = m[0, 0], m[1, 1], m[2, 2]
    zmax = origin[2] + Lz
    z0, z1 = zrange if zrange else (0.0, 1.2 * zmax)

    W = int(width)
    H = max(64, int(round(W * Lx / (z1 - z0))))

    pipeline.source.data.particles_.vis.radius = radius
    pipeline.source.data.cell_.vis.enabled = False   # no per-rank box outline
    pipeline.modifiers.append(ColorCodingModifier(
        property=prop, start_value=crange[0], end_value=crange[1],
        gradient=ColorCodingModifier.Rainbow()))
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Ortho)
    vp.camera_dir = (0.0, 1.0, 0.0)         # look along +y  -> image right = +z, up = +x
    vp.camera_up = (1.0, 0.0, 0.0)
    vp.camera_pos = (origin[0] + Lx / 2.0,
                     origin[1] - 10.0 * max(Lx, Lz),
                     0.5 * (z0 + z1))
    vp.fov = Lx / 2.0                       # ortho: half the vertical (x) extent, in box units

    vp.render_image(size=(W, H), filename=out_path, alpha=True,
                    background=(0, 0, 0), renderer=TachyonRenderer())
    pipeline.remove_from_scene()
    return W, H, n


def main():
    ap = argparse.ArgumentParser(description="render one LAMMPS dump to an XZ image")
    ap.add_argument("dump", help="LAMMPS dump file (one MPI rank's atoms)")
    ap.add_argument("-o", "--out", default="view1_XZ.png")
    ap.add_argument("--width", type=int, default=5000, help="image width in px (z is horizontal)")
    ap.add_argument("--prop", default="c_PE_All")
    ap.add_argument("--range", nargs=2, type=float, default=[-9.0, -6.5], metavar=("LO", "HI"))
    ap.add_argument("--zrange", nargs=2, type=float, default=None, metavar=("Z0", "Z1"),
                    help="fixed horizontal z-range; default 0 .. 1.2*zmax")
    ap.add_argument("--radius", type=float, default=0.5)
    a = ap.parse_args()

    w, h, n = render_xz(a.dump, a.out, a.prop, tuple(a.range),
                        a.width, tuple(a.zrange) if a.zrange else None, a.radius)
    print(f"wrote {a.out}  {w}x{h} px  from {n} atoms")


if __name__ == "__main__":
    main()
