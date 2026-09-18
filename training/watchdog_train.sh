#!/bin/bash
# Watchdog: keeps the fire/smoke YOLOX-nano training run alive across crashes.
# Restarts with --resume (loads latest_ckpt.pth + optimizer state, continues from the
# last completed epoch) rather than restarting from scratch. On a CUDA OOM specifically,
# drops the batch size one notch before resuming.
set -u
cd /e/fire-and-boundary-detection/training/YOLOX

EXP=exps/custom/fire_nano.py
CKPT=A:/fsbd_training/checkpoints/yolox_nano.pth
LOG=/a/fsbd_training/train.log
LATEST_CKPT=YOLOX_outputs/fire_nano/latest_ckpt.pth
MAX_EPOCH=40
BATCH=8
attempt=0

is_done() {
  # Don't trust the log's "Training of experiment is done" line alone - trainer.py's
  # train() wraps the whole loop in try/except/finally, so that message prints even
  # after a crash mid-training (see fire_nano.py's top-of-file comment for the
  # incident this caused). Verify against the checkpoint's actual stored epoch instead.
  if [ ! -f "$LATEST_CKPT" ]; then
    return 1
  fi
  python -c "
import torch
ck = torch.load(r'$LATEST_CKPT', map_location='cpu', weights_only=False)
import sys
sys.exit(0 if ck.get('start_epoch', 0) >= $MAX_EPOCH else 1)
" 2>/dev/null
}

while true; do
  attempt=$((attempt + 1))

  if is_done; then
    echo "[watchdog] training already complete, exiting." >> /a/fsbd_training/watchdog.log
    break
  fi

  if [ "$attempt" -eq 1 ] && [ -n "${WATCH_PID:-}" ] && kill -0 "$WATCH_PID" 2>/dev/null; then
    echo "[watchdog] attempt 1: an existing training process (PID $WATCH_PID) is already running - waiting for it instead of launching a duplicate." >> /a/fsbd_training/watchdog.log
    while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 5; done
    echo "[watchdog] existing process $WATCH_PID exited." >> /a/fsbd_training/watchdog.log
    if is_done; then
      echo "[watchdog] training complete." >> /a/fsbd_training/watchdog.log
      break
    fi
    echo "[watchdog] not complete yet, entering retry loop." >> /a/fsbd_training/watchdog.log
    continue
  fi

  if [ -f "$LATEST_CKPT" ]; then
    # IMPORTANT: do NOT pass -c here. trainer.py's resume_train() uses args.ckpt as the
    # RESUME checkpoint path when --resume is set (not just for the initial fine-tune
    # load), so passing the original COCO yolox_nano.pth here makes it try to strict-load
    # an 80-class head into this 2-class model and crash. Omitting -c makes it correctly
    # default to <output_dir>/latest_ckpt.pth instead.
    echo "[watchdog] attempt $attempt: resuming from $LATEST_CKPT (batch=$BATCH)" >> /a/fsbd_training/watchdog.log
    python tools/train.py -f "$EXP" -d 1 -b "$BATCH" --fp16 --resume >> "$LOG" 2>&1
  else
    echo "[watchdog] attempt $attempt: fresh start (batch=$BATCH)" >> /a/fsbd_training/watchdog.log
    python tools/train.py -f "$EXP" -d 1 -b "$BATCH" -c "$CKPT" --fp16 >> "$LOG" 2>&1
  fi

  exit_code=$?
  echo "[watchdog] attempt $attempt exited with code $exit_code" >> /a/fsbd_training/watchdog.log

  if is_done; then
    echo "[watchdog] training complete after attempt $attempt." >> /a/fsbd_training/watchdog.log
    break
  fi

  if tail -c 5000 "$LOG" | grep -qi "out of memory\|CUDA error\|CUBLAS_STATUS"; then
    if [ "$BATCH" -gt 2 ]; then
      BATCH=$((BATCH / 2))
      echo "[watchdog] CUDA OOM detected, dropping batch size to $BATCH" >> /a/fsbd_training/watchdog.log
    fi
  fi

  echo "[watchdog] crashed/stopped before completion, retrying in 10s..." >> /a/fsbd_training/watchdog.log
  sleep 10
done
