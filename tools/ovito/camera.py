"""Deterministic cameras for the per-rank OVITO render driver.

Every rank of a frame must build the *identical* camera so the partial
images composite pixel-for-pixel.  A camera is therefore a pure function
of the global simulation box plus a fixed View spec -- never any
per-rank input.

`camera_params()` is pure (no OVITO import) so it can be unit-tested
anywhere; `make_viewport()` is the thin OVITO-touching wrapper.
"""

from dataclasses import dataclass

_AXES = "xyz"


@dataclass(frozen=True)
class View:
    name: str
    look: tuple          # camera view direction (into the scene)
    up: tuple            # world dir mapped to image +y (vertical)
    horiz_axis: int      # box axis shown along the image horizontal (0=x 1=y 2=z)
    vert_axis: int       # box axis shown along the image vertical
    width_px: int = 5000
    horiz_lo: float = 0.0     # horizontal range low edge (world units)
    horiz_factor: float = 1.2  # horizontal range high edge = factor * box_hi[horiz_axis]

    @property
    def depth_axis(self) -> int:
        return 3 - self.horiz_axis - self.vert_axis


# shock runs: long along z -> z is the image horizontal
VIEW1_XZ = View("view1_XZ", look=(0.0, 1.0, 0.0), up=(1.0, 0.0, 0.0),
                horiz_axis=2, vert_axis=0)
VIEW2_YZ = View("view2_YZ", look=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
                horiz_axis=2, vert_axis=1)

VIEWS = {v.name: v for v in (VIEW1_XZ, VIEW2_YZ)}


def box_span(box_matrix):
    """(origin, lengths) from OVITO's 3x4 cell matrix (or any [3][4] seq)."""
    origin = tuple(float(box_matrix[i][3]) for i in range(3))
    lengths = tuple(float(box_matrix[i][i]) for i in range(3))
    return origin, lengths


def camera_params(view: View, box_matrix, horiz_range=None) -> dict:
    """Pure camera math for one view of one global box.

    horiz_range = (h0, h1) overrides the [horiz_lo, factor*hi] default
    for the horizontal axis -- use it to lock image dimensions while the
    box grows (extend_sim).
    """
    origin, lengths = box_span(box_matrix)
    ha, va, da = view.horiz_axis, view.vert_axis, view.depth_axis

    if horiz_range is not None:
        h0, h1 = float(horiz_range[0]), float(horiz_range[1])
    else:
        h0 = view.horiz_lo
        h1 = view.horiz_factor * (origin[ha] + lengths[ha])
    hspan = h1 - h0
    vspan = lengths[va]
    if hspan <= 0.0 or vspan <= 0.0:
        raise ValueError(f"degenerate view extents: hspan={hspan}, vspan={vspan}")

    W = int(view.width_px)
    H = max(64, int(round(W * vspan / hspan)))

    pos = [0.0, 0.0, 0.0]
    pos[ha] = h0 + 0.5 * hspan
    pos[va] = origin[va] + 0.5 * vspan
    pos[da] = origin[da] - 10.0 * max(lengths)     # far back along the depth axis

    return {
        "size": (W, H),
        "camera_dir": tuple(float(c) for c in view.look),
        "camera_up": tuple(float(c) for c in view.up),
        "camera_pos": tuple(pos),
        "fov": 0.5 * vspan,                         # ortho: half the vertical extent
        "horiz": (h0, h1),
        "vert": (pos[va] - 0.5 * vspan, pos[va] + 0.5 * vspan),
    }


def make_viewport(params: dict):
    """Build an OVITO orthographic Viewport from camera_params() output."""
    from ovito.vis import Viewport
    vp = Viewport(type=Viewport.Type.Ortho)
    vp.camera_dir = params["camera_dir"]
    vp.camera_up = params["camera_up"]
    vp.camera_pos = params["camera_pos"]
    vp.fov = params["fov"]
    return vp
