#!/usr/bin/env bash

for arg in "$@"
do
  echo $arg
  sh run_10times.sh $arg
done

