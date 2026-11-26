#!/bin/zsh
# 요청 제한이 풀릴 때까지 기다렸다가 천천히 받고, 다시 막히면 쉬었다가 이어서 받는다.
cd "$(dirname "$0")/.."
LOG=data/raw/download_images.log
for i in {1..30}; do
  echo "[loop] try $i $(date '+%H:%M')" >> $LOG
  TQDM_DISABLE=1 .venv/bin/python scripts/download_images.py --workers 4 --delay 0.5 2>&1 \
    | grep --line-buffered -E "^\[progress\]|^done|errors|403|요청 제한|카나리|Traceback" >> $LOG
  rc=${pipestatus[1]}
  if [[ $rc -eq 0 ]]; then echo "[loop] finished" >> $LOG; exit 0; fi
  if [[ $rc -eq 2 ]]; then echo "[loop] 403, stop" >> $LOG; exit 2; fi
  echo "[loop] rc=$rc, wait 20 min" >> $LOG
  sleep 1200
done
exit 1
