#!/bin/bash
# worker $1 of 2: renders runs $1, $1+2, ... up to $2 (exclusive)
export GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2 GL=llvmpipe SHADOWS=1
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${BETAMAX_RUNS:-$HERE/runs}; mkdir -p $OUT $HERE/logs
for ((i=$1; i<$2; i+=2)); do
  [ -f $OUT/betamax${i}_events.json ] && continue
  for attempt in 1 2 3; do
    timeout 7200 xvfb-run -a -s "-screen 0 640x480x24" node $HERE/run.mjs $i 3000 $OUT >> $HERE/logs/w$1.txt 2>&1 && break
    echo "run $i attempt $attempt failed" >> $HERE/logs/w$1.txt
  done
done
echo "worker $1 finished" >> $HERE/logs/w$1.txt
