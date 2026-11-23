"""Traffic sign predictor for a small device. Needs ONLY: onnxruntime, numpy, Pillow (no PyTorch!).

Usage:
    python predict_sign.py path/to/image.ppm [more images ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent              # the deploy/ folder, wherever it is copied to

# 1. Load the model and the settings saved during training
with open(HERE / "labels.json") as f:
    config = json.load(f)
CLASS_NAMES = config["class_names"]
IMG_SIZE = config["img_size"]
MEAN = np.array(config["mean"], dtype=np.float32).reshape(3, 1, 1)   # shape [3, 1, 1] -> broadcasts over H, W
STD = np.array(config["std"], dtype=np.float32).reshape(3, 1, 1)

options = ort.SessionOptions()
options.intra_op_num_threads = 1
session = ort.InferenceSession(str(HERE / "traffic_sign_int8.onnx"), options,
                               providers=["CPUExecutionProvider"])


def preprocess(image_path):
    """EXACTLY the same steps as in training: RGB -> resize 32x32 -> 0..1 -> normalize."""
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0      # [32, 32, 3], values 0..1
    x = x.transpose(2, 0, 1)                           # -> [3, 32, 32] (channels first, like PyTorch)
    x = (x - MEAN) / STD
    return x[np.newaxis]                               # -> [1, 3, 32, 32] (add batch dimension)


def softmax(logits):
    e = np.exp(logits - logits.max())                  # minus max: avoids overflow
    return e / e.sum()


def predict(image_path):
    x = preprocess(image_path)
    start = time.perf_counter()
    logits = session.run(None, {"image": x})[0][0]     # [43]
    ms = (time.perf_counter() - start) * 1000
    probs = softmax(logits)
    best = int(probs.argmax())
    return CLASS_NAMES[best], float(probs[best]), ms


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        name, confidence, ms = predict(path)
        print(f"{Path(path).name}: {name}  (confidence {confidence:.0%}, {ms:.2f} ms)")
