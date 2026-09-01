"""Tests for cull.visible() -- synthetic per-rank dump files, no OVITO.

    python test_cull.py
"""

import os
import tempfile

from camera import VIEW1_XZ, camera_params
from cull import visible

BOX = [[20.0, 0, 0, 0], [0, 10.0, 0, 0], [0, 0, 100.0, 0]]  # x0..20 y0..10 z0..100


def write_dump(path, atoms):
    with open(path, "w") as fh:
        fh.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n" % len(atoms))
        fh.write("ITEM: BOX BOUNDS pp pp pp\n0 20\n0 10\n0 100\n")
        fh.write("ITEM: ATOMS id type x y z c_PE_All\n")
        for i, (x, y, z) in enumerate(atoms):
            fh.write(f"{i+1} 1 {x} {y} {z} -7.0\n")


def build(tmp, spec):
    files = []
    for i, atoms in enumerate(spec):
        p = os.path.join(tmp, f"dump.{i}.0")
        write_dump(p, atoms)
        files.append(p)
    return files


def test_empty_and_frustum():
    with tempfile.TemporaryDirectory() as tmp:
        files = build(tmp, [
            [(10, 5, 10)],                     # in view
            [],                                # empty
            [(10, 5, 500)],                    # z=500 -> above 1.2*100=120, frustum
        ])
        params = camera_params(VIEW1_XZ, BOX)
        keep, st = visible(files, VIEW1_XZ, params)
        assert st == {"empty": 1, "frustum": 1, "occluded": 0, "kept": 1}, st
        assert keep == [files[0]]


def test_occlusion_y_split():
    # three ranks, same x,z footprint, stacked in y (depth for view1_XZ)
    with tempfile.TemporaryDirectory() as tmp:
        files = build(tmp, [
            [(2, 1, 2), (18, 2, 98)],          # front  y ~ 1..2
            [(2, 5, 2), (18, 6, 98)],          # middle y ~ 5..6
            [(2, 9, 2), (18, 9, 98)],          # back   y ~ 9
        ])
        params = camera_params(VIEW1_XZ, BOX)
        keep, st = visible(files, VIEW1_XZ, params, occlude=True, pad=1.0)
        assert st["occluded"] == 2, st
        assert keep == [files[0]]
        # default (no occlude) keeps all three
        keep2, st2 = visible(files, VIEW1_XZ, params)
        assert st2["kept"] == 3


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
    print("passed")
