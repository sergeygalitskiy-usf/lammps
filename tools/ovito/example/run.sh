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
"$OVITO" "$DRIVER" dump/ -o 'frames/{view}.{step}.png' \
    --views view1_XZ,view2_YZ --prop c_PE_All --range -7 -5 \
    --width 3000 --nproc 4 --interval 2 --idle 30 --until 4000 &
WATCH=$!

mpirun -np 4 "$LMP" -in in.shock

wait "$WATCH"

# odd dimensions -> pad to even for libx264
for v in view1_XZ view2_YZ; do
    ffmpeg -y -framerate 8 -pattern_type glob -i "frames/${v}.*.png" \
        -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "${v}.mp4"
    echo "wrote ${v}.mp4"
done
