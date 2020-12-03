#!/usr/bin/env bash

# 同步整个das目录到集群4个机器
rsync -r ../das experiment@210.28.132.11:./huqiu/

for i in {0..9}
do
ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/; rsync -r ./das slave02${i}:/home/experiment/huqiu/"
echo "slave02${i} sync done"
# ssh experiment@210.28.132.11 "cd /home/experiment/huqiu/; rsync -r ./das slave025:/home/experiment/huqiu/"
done
