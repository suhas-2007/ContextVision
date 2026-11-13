# ContextVision

ContextVision detects everyday objects and scores how well each detection fits its surroundings. It uses YOLO11s-OBB, CLIP ViT-B/32, a trained context selector, and Hopfield memory.

## Run

Install the Python dependencies and place the supplied `Hopfield networks.zip` next to `context_anomaly.py`. The archive contains the detector and selector weights, the context memory, and the validation images. CLIP weights download on first use.

```sh
python -m pip install torch torchvision ultralytics openai-clip
python context_anomaly.py
python context_anomaly.py --evaluate
```

The script saves annotated images and JSON results in `context_results/`. Pass `--image path/to/image.jpg` to try your own image.

## Validation

On 20 validation images at input size 256, the detector matched 16 of 22 labelled objects using same-class polygon IoU of at least 0.50. The context stage flagged 7 detections at the 0.90 similarity cutoff. The validation labels cover classes and boxes. Anomaly accuracy was not measured.

See [the project report](Context_Anomaly_Project_Report.pdf) for the model and experiment details.
