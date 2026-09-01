"""Unit tests for camera_params() -- pure math, no OVITO needed.

    python test_camera.py          # or: ovitos test_camera.py
"""

from camera import VIEW1_XZ, VIEW2_YZ, camera_params

# global box: x 0..20, y 0..10, z 0..100  (OVITO 3x4 cell layout)
BOX = [[20.0, 0.0, 0.0, 0.0],
       [0.0, 10.0, 0.0, 0.0],
       [0.0, 0.0, 100.0, 0.0]]


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_view1_xz_sizing():
    p = camera_params(VIEW1_XZ, BOX)
    W, H = p["size"]
    assert W == 5000
    # horizontal = z: 0 .. 1.2*100 = 120 ; vertical = x: 20 ; H = 5000*20/120
    assert H == round(5000 * 20.0 / 120.0), H
    assert p["horiz"] == (0.0, 120.0)
    assert approx(p["fov"], 10.0)                    # half the x extent
    assert p["camera_dir"] == (0.0, 1.0, 0.0)
    assert p["camera_up"] == (1.0, 0.0, 0.0)
    # centre: x mid=10, z mid=60, y far back
    assert approx(p["camera_pos"][0], 10.0)
    assert approx(p["camera_pos"][2], 60.0)
    assert p["camera_pos"][1] < 0.0


def test_zrange_override_locks_dims():
    p = camera_params(VIEW1_XZ, BOX, horiz_range=(0.0, 250.0))
    assert p["horiz"] == (0.0, 250.0)
    assert p["size"][1] == round(5000 * 20.0 / 250.0)


def test_view2_yz():
    p = camera_params(VIEW2_YZ, BOX)
    # horizontal = z (120), vertical = y (10)
    assert p["size"][1] == round(5000 * 10.0 / 120.0)
    assert approx(p["fov"], 5.0)
    assert p["camera_dir"] == (1.0, 0.0, 0.0)
    assert p["camera_pos"][0] < 0.0                  # far back along x (depth axis)


def test_deterministic():
    a = camera_params(VIEW1_XZ, BOX)
    b = camera_params(VIEW1_XZ, [list(r) for r in BOX])
    assert a == b


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
