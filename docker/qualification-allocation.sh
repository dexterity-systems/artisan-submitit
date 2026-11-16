#!/bin/bash
# Run a controller inside the fixture's default-config allocation.
set -euo pipefail
[[ ${SLURM_JOB_ID-} =~ ^[0-9]+$ ]] || exit 64
[[ ${SLURM_CONF-} == /etc/slurm/slurm.conf ]] || exit 64
unset SLURM_CONF
status=0
"$@" || status=$?
/usr/bin/scontrol show job "$SLURM_JOB_ID"
exit "$status"
