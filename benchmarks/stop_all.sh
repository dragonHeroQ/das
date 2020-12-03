#!/usr/bin/env bash

DAS_HOME=/home/experiment/huqiu/das/benchmarks/

if [[ -f ${DAS_HOME}/slaves ]]; then
{
  for line in $(cat ${DAS_HOME}/slaves)
  do
  {
    echo "${line} ray stop "
    ssh ${line} "ray stop"
  }
  done
}
fi

echo "stop master"
ray stop
