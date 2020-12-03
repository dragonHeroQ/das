#!/usr/bin/env bash

data=$1
echo ${data}
for i in {0..9}
do
# python t_grid_val_process.py ${data} ${i} |& tee ${data}_${i}_filterphase.log
python exp_compmodelselection.py ${data} ${i} 7200 |& tee ${data}_${i}_combination_7200.log
done
