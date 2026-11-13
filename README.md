# ContextVision

ContextVision looks for everyday objects that might be out of place in a scene. It detects screwdrivers, combs, knives, pens and toothbrushes.

## How it works

1. YOLO11s-OBB predicts a class and a rotated box.
2. CLIP embeds four views around the box. A trained attention model picks the view to use. For one detection, the program uses the full image.
3. A Hopfield memory retrieves vectors for the selected view and the class name. The script flags a box when their cosine similarity is 0.90 or lower.

See the [flowchart](docs/workflow.md) and [project report](Context_Anomaly_Project_Report.pdf) for details.

## Dataset and setup

The annotated dataset has 531 training images and 20 validation images. Download the [project archive and dataset](https://github.com/suhas-2007/ContextVision/releases/tag/project-files). It includes the detector, context selector and memory bank. The script reads these files directly from the ZIP.

Run these commands from the repository folder. CLIP weights download on the first run.

```sh
python -m pip install -r requirements.txt
python context_anomaly.py --archive ContextVision-project-files.zip
python context_anomaly.py --archive ContextVision-project-files.zip --evaluate
```

Pass `--image path/to/image.jpg` to try your own image. The script saves annotated images and scores in `context_results/`.

## Results

| Detector input size | Matched boxes out of 22 | Precision | Recall | Flagged detections |
| --- | ---: | ---: | ---: | ---: |
| 256 | 16 | 72.7% | 72.7% | 7 |
| 896 | 3 | 7.7% | 13.6% | 19 |

A match required the correct class and at least 0.50 overlap by rotated polygon IoU. Both runs used a confidence threshold of 0.25. The validation set has object labels only, so the flag count does not measure anomaly accuracy.

[Example output](reports/example-output.jpg) · [256 run](reports/validation-256.json) · [896 run](reports/validation-896.json)

## Team

Spurjan, Madhav and Suhas
