# ContextVision

ContextVision looks for everyday objects that might be out of place in a scene. It detects screwdrivers, combs, knives, pens and toothbrushes.

## How it works

1. YOLO11s-OBB predicts a class and a rotated box.
2. CLIP embeds four views around the box. A trained attention model picks the view to use. For one detection, the program uses the full image.
3. A Hopfield memory retrieves vectors for the selected view and the class name. The script flags a box when their cosine similarity is 0.90 or lower.

See the [flowchart](docs/workflow.md) and [project report](Context_Anomaly_Project_Report.pdf) for details.

## Dataset

The annotated dataset has 531 training images and 20 validation images. The [dataset release](https://github.com/suhas-2007/ContextVision/releases/tag/project-files) also includes the trained detector, context selector and memory bank.

## How to run it

1. Clone the repo and open its folder in a terminal.

   ```sh
   git clone https://github.com/suhas-2007/ContextVision.git
   cd ContextVision
   ```

2. Download [ContextVision-project-files.zip](https://github.com/suhas-2007/ContextVision/releases/download/project-files/ContextVision-project-files.zip) and put it next to `context_anomaly.py`. Leave it zipped.

3. Install the packages.

   ```sh
   python -m pip install -r requirements.txt
   ```

4. Run the demo. The first run downloads the CLIP weights, so it takes a bit longer.

   ```sh
   python context_anomaly.py --archive ContextVision-project-files.zip
   ```

Open `context_results/demo.jpg` to see the boxes and `context_results/demo.json` to read the scores.

To try your own image or rerun validation, use one of these commands:

```sh
python context_anomaly.py --archive ContextVision-project-files.zip --image "path/to/your-image.jpg"
python context_anomaly.py --archive ContextVision-project-files.zip --evaluate
```

Replace the example image path with your file. Validation writes `context_results/validation.json` and an annotated image for each input.

## Results

| Detector input size | Matched boxes out of 22 | Precision | Recall | Flagged detections |
| --- | ---: | ---: | ---: | ---: |
| 256 | 16 | 72.7% | 72.7% | 7 |
| 896 | 3 | 7.7% | 13.6% | 19 |

A match required the correct class and at least 0.50 overlap by rotated polygon IoU. Both runs used a confidence threshold of 0.25. The validation set has object labels only, so the flag count does not measure anomaly accuracy.

[Example output](reports/example-output.jpg) · [256 run](reports/validation-256.json) · [896 run](reports/validation-896.json)

## Team

Spurjan, Madhav and Suhas
