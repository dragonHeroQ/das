#!/usr/bin/env bash

for arg in "$@"
do
  echo $arg
  sh run_5times_with_ucb.sh $arg
done
