#!/bin/bash
# Watchdog: keeps the PPE YOLOX-nano training run alive across crashes.
# Same shape as watchdog_train.sh (fire/smoke), adapted for the PPE exp - see that
# script's comments for the two known-fixed failure modes this mirrors (in-training
# eval crash on this machine's missing MSVC toolchain, and --resume/-c interaction).
# Path below is corrected for this repo's current name (was fire-and-boundary-detection
# during the fire/smoke run, renamed to multi-modal-detection since).
set -u
cd /e/multi-modal-detection/training/YOLOX

EXP=exps/custom/ppe_nano.py
CKPT=A:/fsbd_training/checkpoints/yolox_nano.pth
LOG=/a/fsbd_training/ppe_train.log
LATEST_CKPT=YOLOX_outputs/ppe_nano/latest_ckpt.pth
MAX_EPOCH=40
BATCH=8
attempt=0

is_done() {
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
    echo "[watchdog] training already complete, exiting." >> /a/fsbd_training/ppe_watchdog.log
    break
  fi

  if [ -f "$LATEST_CKPT" ]; then
    # do NOT pass -c on resume - see watchdog_train.sh's comment, same trainer.py behavior.
    echo "[watchdog] attempt $attempt: resuming from $LATEST_CKPT (batch=$BATCH)" >> /a/fsbd_training/ppe_watchdog.log
    python tools/train.py -f "$EXP" -d 1 -b "$BATCH" --fp16 --resume >> "$LOG" 2>&1
  else
    echo "[watchdog] attempt $attempt: fresh start (batch=$BATCH)" >> /a/fsbd_training/ppe_watchdog.log
    python tools/train.py -f "$EXP" -d 1 -b "$BATCH" -c "$CKPT" --fp16 >> "$LOG" 2>&1
  fi

  exit_code=$?
  echo "[watchdog] attempt $attempt exited with code $exit_code" >> /a/fsbd_training/ppe_watchdog.log

  if is_done; then
    echo "[watchdog] training complete after attempt $attempt." >> /a/fsbd_training/ppe_watchdog.log
    break
  fi

  if tail -c 5000 "$LOG" | grep -qi "out of memory\|CUDA error\|CUBLAS_STATUS"; then
    if [ "$BATCH" -gt 2 ]; then
      BATCH=$((BATCH / 2))
      echo "[watchdog] CUDA OOM detected, dropping batch size to $BATCH" >> /a/fsbd_training/ppe_watchdog.log
    fi
  fi

  echo "[watchdog] crashed/stopped before completion, retrying in 10s..." >> /a/fsbd_training/ppe_watchdog.log
  sleep 10
done
