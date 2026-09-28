# PPE helmet demo (CPU)

Detects people, then checks each person's head: **HELMET** (green) or **NO HELMET** (red, shown after
a bare head is seen in 3 of the last 5 frames). CPU only; no GPU needed.

```
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python demo.py --source samples/test_input.mp4
```

Other sources: `--source 0` (webcam), `--source rtsp://user:pass@host/stream`.
Save the annotated video: `--save out.mp4`. Headless: `--no-show`. Quit: `q`.

`samples/test_input.mp4` is a slideshow of held-out test images (not CCTV footage);
`samples/annotated_output.mp4` is the demo's output on it. Metrics and limits: `model_card.md`.
