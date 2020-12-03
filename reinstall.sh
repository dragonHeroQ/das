#!/usr/bin/env bash
# 同步源代码更新(./das/)到集群4个机器
rsync -r ./das experiment@210.28.132.11:./huqiu/das/
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/; rsync -r ./das slave023:/home/experiment/huqiu/das/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/; rsync -r ./das slave025:/home/experiment/huqiu/das/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/; rsync -r ./das slave024:/home/experiment/huqiu/das/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/; rsync -r ./das slave026:/home/experiment/huqiu/das/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/; rsync -r ./das slave027:/home/experiment/huqiu/das/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/; rsync -r ./das slave028:/home/experiment/huqiu/das/"
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/das/; rsync -r ./das slave029:/home/experiment/huqiu/das/"
