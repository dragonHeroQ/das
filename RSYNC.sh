#!/usr/bin/env bash
# 同步benchmarks到集群4个机器
rsync ./benchmarks/*.py experiment@210.28.132.11:./huqiu/das/benchmarks/
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.py slave023:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.py slave025:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.py slave024:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.py slave026:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.py slave027:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.py slave028:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.py slave029:/home/experiment/huqiu/das/benchmarks/"

rsync ./benchmarks/*.sh experiment@210.28.132.11:./huqiu/das/benchmarks/
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.sh slave023:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.sh slave025:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.sh slave024:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.sh slave026:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.sh slave027:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.sh slave028:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync *.sh slave029:/home/experiment/huqiu/das/benchmarks/"

rsync ./benchmarks/slaves experiment@210.28.132.11:./huqiu/das/benchmarks/
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync slaves slave023:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync slaves slave025:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync slaves slave024:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync slaves slave026:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync slaves slave027:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync slaves slave028:/home/experiment/huqiu/das/benchmarks/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/benchmarks; rsync slaves slave029:/home/experiment/huqiu/das/benchmarks/"

#rsync -r ./benchmarks experiment@210.28.132.11:./huqiu/das/
#ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das; rsync -r benchmarks slave023:/home/experiment/huqiu/das/"
#ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das; rsync -r benchmarks slave025:/home/experiment/huqiu/das/"
#ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das; rsync -r benchmarks slave024:/home/experiment/huqiu/das/"
#ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das; rsync -r benchmarks slave026:/home/experiment/huqiu/das/"
#ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das; rsync -r benchmarks slave027:/home/experiment/huqiu/das/"
#ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das; rsync -r benchmarks slave028:/home/experiment/huqiu/das/"
#ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das; rsync -r benchmarks slave029:/home/experiment/huqiu/das/"
