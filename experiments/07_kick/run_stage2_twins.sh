#!/bin/bash
# stage-2 twins (experiments/07_kick/DESIGN_stage2.md step 4): each command rerun until it exits cleanly (max 6 tries)
cd /workspace/swarm-cavity && source scripts/env.sh
run() { for i in 1 2 3 4 5 6; do uv run python experiments/07_kick/kick.py "$@" 2>&1 | tee -a results/kick2k-20261125.log; rc=${PIPESTATUS[0]}; echo "TRY $i EXIT $rc: $*" | tee -a results/kick2k-20261125.log; [ "$rc" = 0 ] && return 0; sleep 60; done; return 1; }
run kick2-20261125 kick2k-20261125 --t0 6 --rep 0 --min-abs-s 0.1 --cap 6 --parallel 32 --cells K4-pn0.5-g+0.0
run kick2-20261125 kick2k-20261125 --t0 6 8 --rep 0 1 --min-abs-s 0.1 --sham 16 --cap 60 --parallel 32 --cells K2-pn0.5-g-0.5 K2-pn0.5-g+0.5 K4-pn0.5-g-1.0 K4-pn0.5-g+1.0
run kick2-20261125 kick2e-20261125 --t0 0 1 --rep 0 --min-abs-s 0.1 --cap 18 --parallel 32 --cells K4-pn0.5-g-1.0 K4-pn0.5-g+1.0
echo KICK2_ALL_DONE | tee -a results/kick2k-20261125.log
