#!/usr/bin/env bash

scp /home/experiment/huqiu/das/benchmarks/lcvs/lcv_del_list.txt slave023:/home/experiment/huqiu/das/benchmarks/lcvs
ssh slave023 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh del_lcvs.sh"
scp /home/experiment/huqiu/das/benchmarks/lcvs/lcv_del_list.txt slave025:/home/experiment/huqiu/das/benchmarks/lcvs
ssh slave025 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh del_lcvs.sh"
scp /home/experiment/huqiu/das/benchmarks/lcvs/lcv_del_list.txt slave024:/home/experiment/huqiu/das/benchmarks/lcvs
ssh slave024 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh del_lcvs.sh"
scp /home/experiment/huqiu/das/benchmarks/lcvs/lcv_del_list.txt slave026:/home/experiment/huqiu/das/benchmarks/lcvs
ssh slave026 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh del_lcvs.sh"
scp /home/experiment/huqiu/das/benchmarks/lcvs/lcv_del_list.txt slave027:/home/experiment/huqiu/das/benchmarks/lcvs
ssh slave027 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh del_lcvs.sh"
scp /home/experiment/huqiu/das/benchmarks/lcvs/lcv_del_list.txt slave028:/home/experiment/huqiu/das/benchmarks/lcvs
ssh slave028 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh del_lcvs.sh"
scp /home/experiment/huqiu/das/benchmarks/lcvs/lcv_del_list.txt slave029:/home/experiment/huqiu/das/benchmarks/lcvs
ssh slave029 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh del_lcvs.sh"

sh /home/experiment/huqiu/das/benchmarks/cmds/del_lcvs.sh
