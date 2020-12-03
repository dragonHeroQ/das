#!/usr/bin/env bash

LCV_HOME=/home/huqiu/das/benchmarks/lcvs

if [[ -f ${LCV_HOME}/lcv_del_list.txt ]]; then
{
  for line in $(cat ${LCV_HOME}/lcv_del_list.txt)
  do
  {
    rm ${LCV_HOME}/${line}
  }
  done
}
fi
