# Machine Learning & Computer Vision with PyTorch

I come from industrial machine maintenance: I used to fix machines **after** they broke.
Now I build models that see defects, predict failures **before** they happen, and run on small hardware.

This repo holds my projects and my hands-on course notebooks, alongside my M.Eng. in
*Artificial Intelligence for Smart Sensors and Actuators* (Deggendorf Institute of Technology).

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005ced?logo=onnx&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-f37626?logo=jupyter&logoColor=white)

---

## Projects

| Project | Data | Key result |
|---|---|---|
| [🚦 Traffic sign recognition → deployment](#-traffic-sign-recognition-from-training-to-deployment) | GTSRB, 43 classes, 39k images | **98.6%** test accuracy, INT8 model **3.6× smaller**, **0.21 ms** per image on 1 CPU thread |
| [🔧 Bearing fault detection (predictive maintenance)](#-bearing-fault-detection-from-vibration-predictive-maintenance) | CWRU vibration signals | Autoencoder trained on **healthy data only**: detects **100%** of faults with **0.9%** false alarms |
| [🏭 Casting defect detection](#-casting-defect-detection) | 7.3k images of metal castings | **97%** test accuracy with a small CNN |

---

### 🚦 Traffic sign recognition: from training to deployment
📓 [`traffic_signs.ipynb`](traffic_signs.ipynb) · 📦 [`deploy/`](deploy/)

Training a good classifier is the easy part. Car makers care about **size, speed and running without a big framework**. This project goes all the way from data to a standalone app.

![The 43 GTSRB classes](assets/traffic_sign_classes.png)

**What I did**
- **Leak-free validation split by track.** GTSRB films every real sign 30 times from a moving car. A random split puts almost identical photos in both train and validation, so I split by *physical sign* instead.
- **Augmentation that keeps the meaning:** small rotations, shifts and light changes, but **no horizontal flips**, because a flipped "Keep right" *is* "Keep left".
- **A small CNN** (293k parameters, 3 conv blocks + global average pooling) with label smoothing and a cosine learning-rate schedule.
- **Deployment pipeline:** PyTorch → **ONNX** → **INT8 static quantization** (calibrated on training images), measured like an embedded device: 1 CPU thread, batch size 1, with warm-up.
- **A standalone predictor** ([`deploy/predict_sign.py`](deploy/predict_sign.py)) that needs only `onnxruntime`, `numpy` and `Pillow`. The preprocessing settings are saved in `labels.json`, so the device does exactly the same steps as training.

![Deployment results](assets/traffic_deployment.png)

| Version | Size | Latency (1 thread) | Test accuracy |
|---|---|---|---|
| PyTorch FP32 | 1161 KB | 5.45 ms | 98.57% |
| ONNX FP32 | 1175 KB | 0.50 ms | 98.57% |
| **ONNX INT8** | **325 KB** | **0.21 ms** | **98.58%** |

**Limits & next steps:** GTSRB images are already cropped around the sign. A real car first needs a **detector** (e.g. YOLO) to find signs in the full camera frame. Latency was measured on a laptop CPU; the next step is measuring on a Raspberry Pi.

---

### 🔧 Bearing fault detection from vibration (predictive maintenance)
📓 [`bearing_faults_full.ipynb`](bearing_faults_full.ipynb)

An accelerometer on a motor bearing records vibration 12,000 times per second. A crack makes a small "hit" every time a ball rolls over it. The goal is to recognize these patterns **before** the bearing fails.
10 classes: healthy + 3 fault locations (inner race, ball, outer race) × 3 fault sizes.

![Raw vibration signals](assets/bearing_signals.png)

**What I did**
- **Leak-free split by time:** each recording is split 60/20/20 **before** cutting it into windows, so overlapping windows never appear in both train and test.
- **Classic ML baseline:** 26 physics-inspired features (RMS, kurtosis, crest factor, FFT band energies) + Random Forest.
- **1D CNN** on raw signal windows, with a wide first kernel.
- **Autoencoder for anomaly detection**, trained on **healthy data only**, which is the realistic case in a factory, where fault examples are rare. A high reconstruction error raises an alarm.
- **Robustness test** on new recordings at a different motor load (1 HP).

![Autoencoder health monitor](assets/bearing_health_monitor.png)

| Model | Same load (0 HP) | New load (1 HP) | New load, fault *location* only |
|---|---|---|---|
| Random Forest + features | 100% | 86.6% | **99.4%** |
| 1D CNN | 96.3% | 74.9% | 84.2% |

**Autoencoder at the new load:** detects **100%** of faults with **0.9%** false alarms, after calibrating the alarm threshold on a short healthy baseline (no fault data used).

**Problems I found and fixed**
- 🐛 **A hidden data bug:** the healthy recordings are sampled at **48 kHz**, the fault recordings at **12 kHz**. I found it with physics: the shaft frequency (1796 RPM / 60 ≈ 30 Hz) only appears in the right place at 48 kHz. Without resampling, models learn a shortcut ("slow-looking signal = healthy").
- 🎚️ **Alarm threshold:** trained at one load, the autoencoder flagged *every* healthy window at a new load, but healthy (error ≈ 5) and faulty (error ≥ 22) were still clearly separated. The fix is to calibrate the threshold per operating condition, like commissioning a real monitoring system.
- 📉 **Simple beats deep here:** with ~10 s of data per class, physics-based features + Random Forest generalize better to a new load than the CNN.

**Limits:** CWRU faults are artificially made on a lab test rig, and the data shows *fault vs. no fault*, not *time until failure*. The next step is a run-to-failure dataset (IMS, PRONOSTIA) for Remaining Useful Life prediction.

---

### 🏭 Casting defect detection
📓 [`casting_data_full.ipynb`](casting_data_full.ipynb)

Visual quality inspection of metal casting parts (submersible pump impellers): **defective** or **ok**.

![Casting samples](assets/casting_samples.png)

**What I did**
- Loaded the images with `ImageFolder` (grayscale, 64×64) and built a small CNN (2 conv blocks + classifier) in PyTorch.
- Trained it with the standard training loop, tracked loss/accuracy curves, and saved and reloaded the model (`state_dict`).
- Evaluated with a **confusion matrix**, not only accuracy: in a factory, the costly mistake is shipping a **defective** part as "ok".

**Result:** **97.0%** test accuracy; 97.6% of defective parts caught (recall).

---

## Course notebooks

### FashionMNIST: linear baseline vs CNN

| Model | Architecture | Test accuracy |
|---|---|---|
| Linear baseline | Flatten → Linear → Linear | 84.0 % |
| **CNN** | 2 × (Conv-ReLU-Conv-ReLU-MaxPool) → Linear | **87.6 %** (3 epochs, 58 s on GPU) |

<p align="center">
  <img src="assets/fashionmnist_samples.png" width="38%" alt="FashionMNIST samples">
  &nbsp;
  <img src="assets/fashionmnist_confusion_matrix.png" width="42%" alt="CNN confusion matrix">
</p>

The confusion matrix shows where the CNN struggles: **Shirt** is the hardest
class (63 % correct), mostly confused with T-shirt/top, Coat and Pullover.
Trousers, bags, sandals and ankle boots are above 96 %.

→ [`pytorch-course/03_computer_vision_fashionmnist.ipynb`](pytorch-course/03_computer_vision_fashionmnist.ipynb)

### `pytorch-course/`: Zero to Mastery, *Learn PyTorch for Deep Learning*

| Notebook | Topics |
|---|---|
| [00_pytorch_basics](pytorch-course/00_pytorch_basics.ipynb) | Environment check, CUDA, first tensors |
| [03_computer_vision_fashionmnist](pytorch-course/03_computer_vision_fashionmnist.ipynb) | `torchvision` datasets, `DataLoader`, linear model vs CNN, device-agnostic train/test functions, confusion matrix |
| [04_custom_datasets](pytorch-course/04_custom_datasets.ipynb) | Image-folder dataset (pizza / steak / sushi): exploring, transforms, `ImageFolder`, `DataLoader`, and a custom `Dataset` class *(in progress)* |

`helper_functions.py` comes from the course repository.

### `llm-course/`: *A deep understanding of AI large language model mechanisms*

Course code by Mike X Cohen that I'm working through: tokenization and
embeddings, building a GPT-style transformer from scratch, pretraining and
fine-tuning, evaluation, and mechanistic interpretability (hooks, attention
heads, activation patching). See [`llm-course/README.md`](llm-course/README.md).

---

## Running the notebooks

```bash
git clone https://github.com/muemara1/ML.git
cd ML
python -m venv .venv-gpu
# Windows: .venv-gpu\Scripts\activate    Linux/macOS: source .venv-gpu/bin/activate

# For an NVIDIA GPU, install the CUDA build of torch first (see pytorch.org)
pip install -r requirements.txt
jupyter notebook
```

**Data** is not stored in the repo. The course notebooks download their data on first run. For the projects:

| Project | Where to get it | Put it in |
|---|---|---|
| Traffic signs | downloaded automatically by `torchvision.datasets.GTSRB` | `data/gtsrb/` |
| Bearings | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter), files 97–234 (0 HP) and 98–235 (1 HP), drive end, 12k | `data/cwru/<number>.mat` |
| Casting | Kaggle: *"casting product image data for quality inspection"* | `data/casting_data/{train,test}/{def_front,ok_front}/` |

**Try the deployed traffic sign model** (no PyTorch needed):
```bash
pip install onnxruntime numpy pillow
python deploy/predict_sign.py path/to/sign.png
```

> **AMD GPU note:** on ROCm under Windows, MIOpen may fail to compile its BatchNorm kernel.
> The project notebooks set `torch.backends.cudnn.enabled = False` to use PyTorch's built-in kernels instead.

---

## Credits

- **PyTorch course:** Daniel Bourke, [Zero to Mastery: Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)
  ([code](https://github.com/mrdbourke/pytorch-deep-learning), MIT License).
- **LLM course:** This software includes code developed by Mike X Cohen for the
  Udemy course "A deep understanding of AI large language model mechanisms"
  ([udemy.com/course/dulm_x](https://www.udemy.com/course/dulm_x/)), used under
  the MIT License (Attribution Required). See [`llm-course/LICENSE`](llm-course/LICENSE).
- **Datasets:** [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) (Ruhr-Universität Bochum),
  [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) (Case Western Reserve University),
  and the Kaggle casting product dataset.
