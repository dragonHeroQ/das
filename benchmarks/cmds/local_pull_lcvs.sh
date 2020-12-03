#!/usr/bin/env bash

ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks/cmds; sh master01_pull_lcvs.sh"

rsync -r experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks/lcvs ../
# rsync -r experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks/logs ../
