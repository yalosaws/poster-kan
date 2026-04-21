# POSTER-KAN

POSTER-KAN is a lightweight facial expression recognition model built on top of
the POSTER V2 dual-backbone framework. The main change is a compact
landmark-guided classifier tail: `LandmarkGateFusion` uses facial landmarks to
gate the three POSTER token stages, and `KANHead` replaces the heavier original
classification head with a KAN-style lightweight head.

## Acknowledgement

Part of this codebase is adapted from the original POSTER V2 implementation by
Talented-Q. The reused POSTER V2 code remains under its original MIT license
notice, and the POSTER-KAN modifications are released under the same MIT terms.

## Structure

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

## Setup

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

## Distillation

POSTER-KAN can also be fine-tuned with a teacher checkpoint:

```bash
python train_poster_kan.py \
  --data /path/to/RAF-DB \
  --data-type RAF-DB \
  --init-weights /path/to/poster_kan_v1_best.pth \
  --teacher-checkpoint /path/to/teacher_best.pth \
  --distill-lambda 0.25 \
  --distill-temperature 2.0 \
  --epochs 12 \
  --batch-size 128 \
  --lr 5e-7 \
  --poster-kan-new-lr 6e-5 \
  --poster-kan-backbone-lr-scale 1.0
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
