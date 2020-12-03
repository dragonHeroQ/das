#!/usr/bin/env bash


scp ../lcvs/lcv_del_list.txt experiment@210.28.132.11:./huqiu/das/benchmarks/lcvs
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh master01_del_lcvs.sh"

sh WHATBEG_del_lcvs.sh
