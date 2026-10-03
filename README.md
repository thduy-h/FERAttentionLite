# FERAttentionLite

**Lightweight attention-based facial emotion recognition on FER-2013, with a modular real-time inference pipeline.**

FERAttentionLite is a compact CNN trained from scratch for 7-class facial-expression recognition. The model combines convolutional blocks with lightweight **channel attention** and **spatial attention**, then uses global average pooling to keep the classifier small enough for practical CPU inference.

## Results

The final submission includes two FER-2013 models: **M1 FERAttentionLite** and **M2 MobileNetV3-Small transfer learning**.

| Model | Test accuracy | Macro F1 | Parameters | CPU predictor FPS |
|---|---:|---:|---:|---:|
| M1 FERAttentionLite | **68.54%** | **0.6690** | **1,211,033** | **249.2** |
| M2 MobileNetV3-Small | **67.54%** | **0.6591** | **1,525,031** | **85.8** |

CPU predictor FPS was measured locally with the same benchmark command and includes model-specific preprocessing plus inference. Full camera-pipeline FPS is lower because capture, face detection, drawing and display are also included.

## Architecture

```text
48×48 grayscale face
        │
        ▼
Conv 32 → Conv 32 → MaxPool
        │
        ▼
Conv 64 → Conv 64 → MaxPool
        │
        ▼
Conv 128 → Conv 128
        │
        ├── Channel Attention
        └── Spatial Attention
        │
        ▼
MaxPool
        │
        ▼
Conv 256 → Conv 256
        │
        ▼
Global Average Pooling
        │
        ▼
Linear 256 → 128 → 7 classes
```

The seven output classes follow the FER-2013 ordering used by the training notebook:

```text
Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral
```

The model uses **1.21M trainable parameters** and does not use a pretrained backbone.

## Training recipe

The training notebook covers the complete workflow:

- FER-2013 `Training`, `PublicTest`, and `PrivateTest` splits
- mild geometric augmentation
- softened class weighting using square-root inverse frequency
- weighted cross-entropy with label smoothing
- AdamW optimizer
- cosine learning-rate schedule
- checkpoint selection by validation macro F1
- accuracy and macro-F1 evaluation
- confusion matrix and classification report
- high-confidence error inspection
- model export

Recommended notebook location:

```text
notebooks/train.ipynb
```

## Project structure

```text
FERAttentionLite/
├── README.md
├── requirements.txt
├── config.yaml
├── notebooks/
│   └── FERAttentionLite_FER2013.ipynb
├── weights/
│   └── ... local checkpoints (not committed)
├── outputs/
├── src/
│   ├── app.py
│   ├── benchmark.py
│   ├── detector.py
│   ├── model.py
│   ├── predict.py
│   ├── models/
│   │   └── fer_attention_lite.py
│   ├── predictors/
│   │   ├── base.py
│   │   ├── factory.py
│   │   └── fer_attention_lite.py
│   ├── tools/
│   │   └── convert_to_cpu.py
│   └── utils/
└── .gitignore
```

## Local setup

Python 3.9+ is supported. Python 3.11 is a good default for the local inference environment.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\activate     # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the CLI:

```bash
python -m src.app --help
```

## Checkpoint

Trained checkpoints are intentionally not committed to Git.

After training on Kaggle, place the selected checkpoint at:

```text
weights/best_cnn_cpu.pt
```

To create a CPU-safe copy:

```bash
python -m src.tools.convert_to_cpu \
  --input weights/best.pt \
  --output weights/best_cpu.pt
```

The converter validates the checkpoint against the exact FERAttentionLite architecture before saving it.

For M2:

```bash
python -m src.tools.convert_transfer_to_cpu \
  --input weights/best_transfer.pt \
  --output weights/best_transfer_cpu.pt
```


## Run inference

### Image

```bash
python -m src.app \
  --source test.jpg \
  --model cnn \
  --weights weights/best_cpu.pt
```

### Transfer-learning model

```bash
python -m src.app \
  --source test.jpg \
  --model transfer \
  --weights weights/best_transfer_cpu.pt
```

### Video

```bash
python -m src.app \
  --source test.mp4 \
  --model cnn \
  --weights weights/best_cpu.pt
```

### Webcam

```bash
python -m src.app \
  --source 0 \
  --model cnn \
  --weights weights/best_cpu.pt
```

### IP camera

IP Webcam example:

```bash
python -m src.app \
  --source http://192.168.1.15:8080/video \
  --model cnn \
  --weights weights/best_cpu.pt
```

DroidCam-style URLs can be passed in the same way. The camera address is never hard-coded because the phone IP can change between networks.

## CPU benchmark

```bash
python -m src.benchmark \
  --model cnn \
  --weights weights/best_cpu.pt
```

The benchmark reports classifier latency/FPS separately from full camera-pipeline FPS.

## Modular predictor system

The inference application is model-agnostic. `app.py` does not know the model input size, normalization, class order, or architecture. Each predictor owns those details.

To add another model:

1. add the architecture under `src/models/`;
2. create a predictor that implements `BaseEmotionPredictor`;
3. keep that model's preprocessing and label order inside the predictor;
4. register it in `src/predictors/factory.py`.

The detector, UI drawing, image/video handling, and IP-camera pipeline do not need to be rewritten.

## FERAttentionLite preprocessing

The trained CNN expects:

```text
BGR face crop
→ grayscale
→ resize to 48×48
→ float32 / 255
→ normalize: (x - 0.5) / 0.5
→ tensor [1, 1, 48, 48]
```

Inference must use the same preprocessing as training.

## Notes

- FER-2013 is imbalanced; `Disgust` is particularly under-represented.
- Softened class weighting improved macro F1 and reduced over-prediction of the rare class compared with direct inverse-frequency weighting.
- `Fear`, `Sad`, and `Neutral` remain the most confusable group in the final model.
- The repository separates training from local inference: GPU training is done in the notebook, while the local pipeline targets CPU-friendly real-time use.

## Academic use

This repository was developed as an academic deep-learning lab project. FER-2013 is a third-party dataset; obtain and use it under the terms of the dataset source you choose.

---

If you reproduce or extend the model, keep the preprocessing, label order, and checkpoint architecture aligned. Those three details are the most common source of silent inference errors.
