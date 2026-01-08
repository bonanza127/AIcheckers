#!/bin/bash
LOG=/home/techne/aicheckers/logs/auto_train.log
echo "$(date): Waiting for upload to complete..." >> $LOG

# アップロード完了を待つ（9930枚）
while true; do
    COUNT=$(modal volume ls fastprotect-vol train_images/ 2>/dev/null | wc -l)
    echo "$(date): $COUNT / 9930 images uploaded" >> $LOG
    if [ "$COUNT" -ge 9900 ]; then
        echo "$(date): Upload complete! Starting training..." >> $LOG
        break
    fi
    sleep 30
done

# 学習投入
cd /home/techne/aicheckers
modal run scripts/fastprotect_train.py --submit --use-poisoned-anchors >> $LOG 2>&1
echo "$(date): Training job submitted!" >> $LOG
