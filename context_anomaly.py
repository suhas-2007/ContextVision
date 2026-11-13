"""Run object-context verification using the checkpoints in Hopfield networks.zip.

Install: python -m pip install torch torchvision ultralytics openai-clip
Demo:    python context_anomaly.py
Eval:    python context_anomaly.py --evaluate

Keep the supplied archive beside this file or pass its path with --archive.
CLIP ViT-B/32 downloads on first use. Use --clip-model for a local checkpoint.
Use only the supplied, trusted archive, which contains serialized model data.
"""

import argparse
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import clip
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO


class AttentionHead(nn.Module):
    def __init__(self, dim_size=64):
        super().__init__()
        self.key = nn.Linear(dim_size, dim_size)
        self.query = nn.Linear(dim_size, dim_size)
        self.value = nn.Linear(dim_size, dim_size)

    def forward(self, x, y):
        K = self.key(x)
        Q = self.query(y)
        V = self.value(x)
        d_k = K.size(-1)
        attn_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5), dim=-1)
        return torch.matmul(attn_weights, V)

class MHA(nn.Module):
    def __init__(self, num_heads=8, dim_size=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_size // num_heads
        self.MHA = nn.ModuleList([AttentionHead(self.head_dim) for _ in range(num_heads)])
        self.norm = nn.LayerNorm(dim_size)
    
    def forward(self, x, y):
        attn_outs = []
        for i, head in enumerate(self.MHA):
            start = i * self.head_dim
            end = (i + 1) * self.head_dim
            x_slice = x[:, :, start:end]
            y_slice = y[:, :, start:end]
            attn_outs.append(head(x_slice, y_slice))

        out = torch.cat(attn_outs, dim=-1)
        out = self.norm(out + y)
        return out

class MLP(nn.Module):
    def __init__(self, dim_size=512, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(dim_size, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, dim_size, bias=False)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(dim_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm(out + residual)
        return out

class ContextAggregator(nn.Module):
    def __init__(self, dim_size=512):
        super().__init__()
        self.query_projection = nn.Linear(dim_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.out_projection = nn.Sequential(
            nn.Linear(dim_size, dim_size),
            nn.LayerNorm(dim_size),
            nn.GELU()
        )

    def forward(self, x):
        attention_scores = self.query_projection(x)
        weights = self.softmax(attention_scores)
        combined = torch.matmul(weights.transpose(-2,-1), x)
        out = self.out_projection(combined)
        return out

# ------------------ CONTEXT SELECTOR MODEL ------------------

class ContextSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.MHAmoduoles = nn.ModuleList([MHA(num_heads=8, dim_size=512) for _ in range(4)])
        self.mlp = MLP()
        self.aggregator = ContextAggregator(dim_size=512)

    def forward(self, x, y):
        for mha in self.MHAmoduoles:
            result = mha(x, y)
            x = x + result
        x = self.mlp(x)
        final_prediction = self.aggregator(x)
        return final_prediction




def proposals(image, corners):
    height, width = image.shape[:2]
    left, top = np.floor(corners.min(axis=0)).astype(int)
    right, bottom = np.ceil(corners.max(axis=0)).astype(int)
    left, top = np.clip([left, top], [0, 0], [width - 1, height - 1])
    right, bottom = np.clip([right, bottom], [left + 1, top + 1], [width, height])
    dx, dy = right - left, bottom - top
    cx, cy = (left + right) // 2, (top + bottom) // 2
    return {
        "column": image[:, left:right],
        "row": image[top:bottom, :],
        "box2x": image[max(0, cy - dy):cy + dy, max(0, cx - dx):cx + dx],
        "full": image,
    }


def object_crop(image, corners):
    points = np.asarray(corners, dtype=np.float32)
    width = max(2, int(np.linalg.norm(points[0] - points[1])))
    height = max(2, int(np.linalg.norm(points[1] - points[2])))
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(image, transform, (width, height))


class ContextAnomaly:
    def __init__(self, archive, model_dir, clip_source, cache_dir, device, confidence, threshold, imgsz):
        self.device = device
        self.confidence = confidence
        self.threshold = threshold
        self.imgsz = imgsz
        prefix = "Hopfield networks/"
        weights = model_dir / "best.pt"
        weights.write_bytes(archive.read(prefix + "obb/train4/weights/best.pt"))
        self.detector = YOLO(str(weights))
        self.selector = ContextSelector().to(device).eval()
        selector_path = model_dir / "selector.pth"
        selector_path.write_bytes(archive.read(prefix + "hopfield_context_model.pth"))
        self.selector.load_state_dict(torch.load(selector_path, map_location=device, weights_only=True))
        bank = pickle.loads(archive.read(prefix + "median_matrix.pkl"))
        self.memory = torch.as_tensor(bank, dtype=torch.float32, device=device)
        self.encoder, self.preprocess = clip.load(clip_source, device=device, download_root=str(cache_dir))
        self.encoder.eval()
        with torch.inference_mode():
            labels = [self.detector.names[i] for i in range(len(self.detector.names))]
            tokens = clip.tokenize(labels).to(device)
            self.label_memories = self.retrieve(F.normalize(self.encoder.encode_text(tokens).float()))

    def embed(self, crops):
        inputs = torch.stack([self.preprocess(Image.fromarray(crop)) for crop in crops]).to(self.device)
        return F.normalize(self.encoder.encode_image(inputs).float())

    def retrieve(self, queries):
        return torch.softmax(8 * (queries @ self.memory.T), dim=-1) @ self.memory

    @torch.inference_mode()
    def predict(self, image):
        result = self.detector.predict(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), imgsz=self.imgsz,
            conf=self.confidence, device=self.device, verbose=False,
        )[0]
        detections = result.obb
        if detections is None or len(detections) == 0:
            return []

        rows = []
        for corners, category, confidence in zip(
            detections.xyxyxyxy.cpu().numpy(), detections.cls.int().tolist(), detections.conf.tolist()
        ):
            if len(detections) == 1:
                name = "full"
                context = self.embed([image])
            else:
                crops = proposals(image, corners)
                embeddings = self.embed([*crops.values(), object_crop(image, corners)])
                candidates, query = embeddings[:4], embeddings[4:]
                prediction = self.selector(candidates[None], query[None]).reshape(-1)
                index = int((candidates @ prediction).argmax())
                name = list(crops)[index]
                context = candidates[index:index + 1]
            score = float(F.cosine_similarity(self.retrieve(context), self.label_memories[category:category + 1])[0])
            rows.append({
                "class_id": category, "label": result.names[category], "confidence": confidence,
                "context": name, "similarity": score, "flagged": score <= self.threshold,
                "corners": corners.tolist(),
            })
        return rows


def read_image(data):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode input image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_predictions(image, rows, output):
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label_positions = []
    for row in rows:
        corners = np.rint(row["corners"]).astype(np.int32)
        color = (35, 35, 220) if row["flagged"] else (45, 160, 45)
        cv2.polylines(canvas, [corners], True, color, 3)
        x, y = corners.min(axis=0)
        label = f"{row['label']} {row['similarity']:.3f}"
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        x = min(max(0, x), max(0, canvas.shape[1] - label_width - 2))
        y = max(20, y - 7)
        while any(abs(y - previous_y) < 20 and x < end and x + label_width > start
                  for start, end, previous_y in label_positions):
            y += 20
        label_positions.append((x, x + label_width, y))
        cv2.putText(canvas, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(canvas, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Could not write {output}")


def match_boxes(predictions, labels):
    """Count same-class matches at polygon IoU >= 0.50, highest confidence first."""
    remaining = list(labels)
    matches = 0
    for pred in sorted(predictions, key=lambda row: row["confidence"], reverse=True):
        corners = cv2.convexHull(np.float32(pred["corners"]))
        best_index, best_iou = None, 0.0
        for index, (class_id, points) in enumerate(remaining):
            if class_id != pred["class_id"]:
                continue
            target = cv2.convexHull(points)
            intersection, _ = cv2.intersectConvexConvex(corners, target)
            union = cv2.contourArea(corners) + cv2.contourArea(target) - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_iou >= 0.5:
            remaining.pop(best_index)
            matches += 1
    return matches


def evaluate(archive, pipeline, output):
    image_dir = "Hopfield networks/dataset/images/val/"
    images = sorted(name for name in archive.namelist() if name.startswith(image_dir) and not name.endswith("/"))
    summary = {"images": [], "ground_truth": 0, "detections": 0, "matches": 0, "flagged": 0}
    for name in images:
        image = read_image(archive.read(name))
        filename = Path(name).stem
        height, width = image.shape[:2]
        label_path = f"Hopfield networks/dataset/labels/val/{filename}.txt"
        labels = []
        for line in archive.read(label_path).decode().splitlines():
            if line.strip():
                category, *coords = line.split()
                points = np.float32(coords).reshape(4, 2) * np.float32([width, height])
                labels.append((int(category), points))
        rows = pipeline.predict(image)
        matches = match_boxes(rows, labels)
        summary["images"].append({"file": Path(name).name, "labels": len(labels),
                                  "matches": matches, "predictions": rows})
        summary["ground_truth"] += len(labels)
        summary["detections"] += len(rows)
        summary["matches"] += matches
        summary["flagged"] += sum(row["flagged"] for row in rows)
        draw_predictions(image, rows, output / f"val_{filename}.jpg")
        print(f"{Path(name).name}  labels={len(labels)} detections={len(rows)} matches={matches}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=Path(__file__).with_name("Hopfield networks.zip"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Use your own image instead of the archived demo.")
    source.add_argument("--evaluate", action="store_true", help="Run the detector validation split.")
    parser.add_argument("--output", type=Path, default=Path("context_results"))
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP ViT-B/32 name or local checkpoint.")
    parser.add_argument("--clip-cache", type=Path, default=Path.home() / ".cache" / "clip")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=256, help="Detector input size.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not -1 <= args.threshold <= 1:
        parser.error("Confidence must be in [0, 1] and similarity threshold in [-1, 1].")
    if args.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive, TemporaryDirectory() as model_dir:
        pipeline = ContextAnomaly(archive, Path(model_dir), args.clip_model, args.clip_cache,
                                  args.device, args.confidence, args.threshold, args.imgsz)
        start = perf_counter()
        if args.evaluate:
            result = evaluate(archive, pipeline, args.output)
        else:
            name = args.image.name if args.image else "overlay-image.jpg"
            data = args.image.read_bytes() if args.image else archive.read(f"Hopfield networks/{name}")
            image = read_image(data)
            predictions = pipeline.predict(image)
            draw_predictions(image, predictions, args.output / "demo.jpg")
            result = {"file": name, "predictions": predictions}
        result.update(device=args.device, imgsz=args.imgsz, confidence=args.confidence, threshold=args.threshold,
                      seconds=round(perf_counter() - start, 3))
        destination = args.output / ("validation.json" if args.evaluate else "demo.json")
        destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()


