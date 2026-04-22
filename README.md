# POSTER-KAN

POSTER-KAN is a lightweight facial expression recognition model built on top of
the POSTER V2 dual-backbone framework. The main change is a compact
landmark-guided classifier tail: `LandmarkGateFusion` uses facial landmarks to
gate the three POSTER token stages, and `KANHead` replaces the heavier original
classification head with a KAN-style lightweight head.

This repository provides the model definition and training scripts for
POSTER-KAN. It also includes an optional prototype-adaptive KAN head variant
(`POSTER-KAN-PA`), which keeps the original POSTER-KAN trunk and only swaps the
classifier head.

## Results

POSTER-KAN is designed to keep the POSTER V2 dual-backbone feature extractor
while making the classifier side lighter and faster. With training-time
distillation, the model reaches a comparable RAF-DB accuracy with a much
smaller inference footprint.

| Model | Dataset | Acc (%) | Params (M) | FLOPs (G) | Latency b1 (ms) | Latency b32 (ms) |
|---|---|---:|---:|---:|---:|---:|
| POSTER V2 | RAF-DB | 92.21 | 45.344 | 8.640 | 8.050 | 19.670 |
| POSTER-KAN v1 | RAF-DB | 90.998 | 29.588 | 6.391 | 7.908 | 15.231 |
| POSTER-KAN v2 distill | RAF-DB | 92.07 | 29.588 | 6.391 | 7.908 | 15.231 |
| POSTER-KAN-PA | RAF-DB | 92.107 | 29.588 | 6.391 | same trunk as POSTER-KAN | same trunk as POSTER-KAN |

On AffectNet-7, the same distillation strategy also gives a small accuracy gain.

| Model | Dataset | Acc (%) |
|---|---|---:|
| POSTER V2 | AffectNet-7 | 67.49 |
| POSTER-KAN v2 distill | AffectNet-7 | 67.600 |

POSTER V2 accuracy follows the original paper. Efficiency numbers are measured
with the profiling and benchmark scripts in this repository.

## Acknowledgements

Part of this codebase is adapted from the original POSTER V2 implementation by
Talented-Q. The reused POSTER V2 code remains under its original MIT license
notice, and the POSTER-KAN modifications are released under the same MIT terms.

## Repository Structure

```text
.
├── configs/
│   └── poster_kan.yaml
├── models/
│   ├── PosterV2_7cls.py
│   ├── ir50.py
│   ├── landmark_tokens.py
│   ├── mobilefacenet.py
│   ├── poster_kan.py
│   ├── poster_kan_pa.py
│   └── vit_model.py
├── optim/
│   └── sam.py
├── train_poster_kan.py
├── requirements.txt
└── LICENSE
```

The model expects the same ImageFolder-style dataset layout used by the original
POSTER V2 training code:

```text
dataset_root/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── valid/
    ├── class_0/
    ├── class_1/
    └── ...
```

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Prepare the pretrained backbone weights expected by the POSTER trunk:

```text
models/pretrain/
├── mobilefacenet_model_best.pth.tar
└── ir50.pth
```

These weights are not bundled here. Place them in `models/pretrain/` before
training or evaluation.

## Training

Example RAF-DB training command:

```bash
python train_poster_kan.py \
  --data /path/to/RAF-DB \
  --data-type RAF-DB \
  --epochs 20 \
  --batch-size 128 \
  --lr 3.5e-5 \
  --poster-kan-new-lr 1.0e-4 \
  --poster-kan-backbone-lr-scale 0.05 \
  --loss balanced-softmax \
  --label-smoothing 0.05
```

Checkpoints are written to `outputs/` by default. You can change that with
`--output-dir`.

To train the prototype-adaptive KAN head variant, keep the same entry point and
switch the model variant:

```bash
python train_poster_kan.py \
  --data /path/to/RAF-DB \
  --data-type RAF-DB \
  --model-variant poster-kan-pa \
  --epochs 20 \
  --batch-size 128 \
  --lr 3.5e-5 \
  --poster-kan-new-lr 1.0e-4 \
  --poster-kan-backbone-lr-scale 0.05 \
  --pa-num-prototypes 7 \
  --pa-prototype-scale-init 0.02 \
  --loss balanced-softmax \
  --label-smoothing 0.05
```

## Distillation Fine-Tuning

POSTER-KAN can also be fine-tuned with a teacher checkpoint:

```bash
python train_poster_kan.py \
  --data /path/to/RAF-DB \
  --data-type RAF-DB \
  --model-variant poster-kan-pa \
  --init-weights /path/to/poster_kan_v1_best.pth \
  --teacher-model-variant poster-kan \
  --teacher-checkpoint /path/to/teacher_best.pth \
  --distill-lambda 0.25 \
  --distill-temperature 2.0 \
  --epochs 16 \
  --batch-size 128 \
  --lr 4e-6 \
  --poster-kan-new-lr 8e-5 \
  --poster-kan-backbone-lr-scale 0.05 \
  --pa-prototype-scale-init 0.02
```

## Evaluation

```bash
python train_poster_kan.py \
  --data /path/to/RAF-DB \
  --data-type RAF-DB \
  --evaluate /path/to/best.pth
```

## License

This project is released under the MIT License. See `LICENSE` for the full
license text and copyright notices.
