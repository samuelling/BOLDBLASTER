#!/bin/bash
# Round 12: fit every model on the Godot build's runs (after /home/claude/bbx/run_gd.sh has rendered and processed them)
cd /home/claude/boldblaster
L=/home/claude/bbx/r12_fit.log
for st in fit nonv1 cst stats strip; do echo "== r8 $st $(date +%T)" >> $L; GAME=halcyon_gd python3 run_round8.py $st >> $L 2>&1; done
export R11_GAME=halcyon_gd R11_OUT=round12 R11_RAW=/home/claude/bbx/runs_gd R11_LABEL='Halcyon (Godot)'
for st in pose nav cog new; do echo "== r12 $st $(date +%T)" >> $L; python3 run_round11.py $st >> $L 2>&1; done
echo "== export $(date +%T)" >> $L
python3 export_round8.py >> $L 2>&1; python3 export_round12.py >> $L 2>&1; python3 diag_round12.py >> $L 2>&1
echo "== done $(date +%T)" >> $L
