# Deprecated Landmark-BA VIO

This directory contains the shelved feature-matching and landmark bundle
adjustment VIO experiments. It includes the SuperPoint/LightGlue frontend,
track construction, loop matching, windowed BA, block-coordinate refinement,
global Schur refinement, and their feature/match/track visualizers.

The active learned VIO work remains in `data_processing/vio/` under the
`vio_vggt_*` modules. Shared IMU, evaluation, plotting, and visualization tools
that operate on current trajectory outputs remain in the parent directory.

These files are retained for reproducibility and reference, but they are not
the current VIO implementation.
