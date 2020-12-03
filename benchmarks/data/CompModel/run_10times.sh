#!/usr/bin/env bash

data=$1
echo ${data}
for i in {1..9}
do
python t_grid_val_process.py ${data} $i |& tee ${data}_$i.log
done
