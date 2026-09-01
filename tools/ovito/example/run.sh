#!/usr/bin/env bash
# On-the-fly rendering demo: run LAMMPS and the OVITO watcher concurrently,
# then assemble a movie.
set -euo pipefail
cd "$(dirname "$0")"

LMP=${LMP:-lmp}                                   # any LAMMPS with SHOCK
OVITO=${OVITO:-/Applications/OVITO.app/Contents/MacOS/ovitos}
DRIVER=../render_driver.py

rm -rf dump frames && mkdir -p dump frames

# watcher first: it polls dump/ and renders each frame as its rank files land
"$OVITO" "$DRIVER" dump/ -o 'frames/view1_XZ.{step}.png' \
    --view view1_XZ --prop c_PE_All --range -7 -5 \
    --width 3000 --nproc 4 --interval 2 --idle 30 --until 4000 &
WATCH=$!

mpirun -np 4 "$LMP" -in in.shock

wait "$WATCH"

# odd dimensions -> pad to even for libx264
ffmpeg -y -framerate 8 -pattern_type glob -i 'frames/view1_XZ.*.png' \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p view1_XZ.mp4
echo "wrote view1_XZ.mp4"
