#!/usr/bin/env bash

echo "start master"
ray start --head --redis-port=6379

DAS_HOME=/home/experiment/huqiu/das/benchmarks/

if [[ -f ${DAS_HOME}/slaves ]]; then
{
  for line in $(cat ${DAS_HOME}/slaves)
  do
  {
    echo "ray start ${line}"
    ssh ${line} "sh /home/experiment/huqiu/das/benchmarks/start_worker.sh"
  }
  done
}
fi
