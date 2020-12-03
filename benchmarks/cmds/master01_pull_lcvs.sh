#!/usr/bin/env bash

ssh slave023 "cd /home/experiment/huqiu/das/benchmarks; rsync -r lcvs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave025 "cd /home/experiment/huqiu/das/benchmarks; rsync -r lcvs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave024 "cd /home/experiment/huqiu/das/benchmarks; rsync -r lcvs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave026 "cd /home/experiment/huqiu/das/benchmarks; rsync -r lcvs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave027 "cd /home/experiment/huqiu/das/benchmarks; rsync -r lcvs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave028 "cd /home/experiment/huqiu/das/benchmarks; rsync -r lcvs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave029 "cd /home/experiment/huqiu/das/benchmarks; rsync -r lcvs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"

ssh slave023 "cd /home/experiment/huqiu/das/benchmarks; rsync -r logs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave025 "cd /home/experiment/huqiu/das/benchmarks; rsync -r logs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave024 "cd /home/experiment/huqiu/das/benchmarks; rsync -r logs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave026 "cd /home/experiment/huqiu/das/benchmarks; rsync -r logs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave027 "cd /home/experiment/huqiu/das/benchmarks; rsync -r logs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave028 "cd /home/experiment/huqiu/das/benchmarks; rsync -r logs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
ssh slave029 "cd /home/experiment/huqiu/das/benchmarks; rsync -r logs experiment@210.28.132.11:/home/experiment/huqiu/das/benchmarks"
