#!/bin/bash
# worker $1 of 2: renders tunnel-Betamax runs $1, $1+2, ... up to $2 (exclusive)
export GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2 GL=llvmpipe SHADOWS=1
HERE=/home/claude/bbx
OUT=/home/claude/bbx/runs_dig2
export PREFIX=betamaxu
for ((i=$1; i<$2; i+=2)); do
  [ -f $OUT/betamaxu${i}_events.json ] && continue
  for attempt in 1 2 3; do
    timeout 7200 xvfb-run -a -s "-screen 0 640x480x24" node $HERE/run_dig.mjs $i 3000 $OUT >> $HERE/logs/e$1.txt 2>&1 && break
    echo "run $i attempt $attempt failed" >> $HERE/logs/e$1.txt
  done
done
echo "worker $1 finished" >> $HERE/logs/e$1.txt
