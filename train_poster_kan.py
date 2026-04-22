from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn import metrics
from thop import profile

from models import POSTER_KAN, POSTER_KAN_PA
from optim import SAM


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"


class BalancedSoftmaxLoss(torch.nn.Module):
    def __init__(self, class_counts: torch.Tensor, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("log_prior", class_counts.float().clamp_min(1).log())
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits + self.log_prior.to(logits.device),
            target,
            label_smoothing=self.label_smoothing,
        )


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float | torch.Tensor, n: int = 1) -> None:
        if isinstance(val, torch.Tensor):
            val = float(val.detach().cpu().item())
        self.val = float(val)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: list[AverageMeter], prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate POSTER-KAN.")
    parser.add_argument("--data", type=str, required=True, help="ImageFolder root containing train/ and valid/.")
    parser.add_argument("--data-type", default="RAF-DB", choices=["RAF-DB", "AffectNet-7", "CAER-S"])
    parser.add_argument(
        "--model-variant",
        default="poster-kan",
        choices=["poster-kan", "poster-kan-pa"],
        help="Student model variant.",
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", default=3.5e-5, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--print-freq", default=30, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--loss", default="balanced-softmax", choices=["ce", "balanced-softmax"])
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--landmark-jitter", default=0.015, type=float)
    parser.add_argument("--poster-kan-hidden-dim", default=256, type=int)
    parser.add_argument("--poster-kan-knots", default=8, type=int)
    parser.add_argument("--poster-kan-new-lr", default=1.0e-4, type=float)
    parser.add_argument("--poster-kan-backbone-lr-scale", default=0.05, type=float)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--init-weights", default=None, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--evaluate", default=None, type=str)
    parser.add_argument("--teacher-checkpoint", default=None, type=str)
    parser.add_argument(
        "--teacher-model-variant",
        default="poster-kan",
        choices=["poster-kan", "poster-kan-pa"],
        help="Teacher model variant used only when distillation is enabled.",
    )
    parser.add_argument("--distill-lambda", default=0.0, type=float)
    parser.add_argument("--distill-temperature", default=2.0, type=float)
    parser.add_argument("--pa-num-prototypes", default=7, type=int)
    parser.add_argument("--pa-prototype-scale-init", default=0.0, type=float)
    parser.add_argument("--pa-no-gate", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--benchmark-speed", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    return parser.parse_args()


def resolve_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.gpu).lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_counts(dataset: datasets.ImageFolder, num_classes: int) -> torch.Tensor:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [label for _, label in dataset.samples]
    return torch.bincount(torch.as_tensor(targets, dtype=torch.long), minlength=num_classes).clamp_min(1)


def build_train_transform(data_type: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if data_type == "RAF-DB":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.12)),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model_variant(variant: str, args: argparse.Namespace) -> torch.nn.Module:
    common_kwargs = {
        "img_size": 224,
        "num_classes": 7,
        "landmark_jitter": args.landmark_jitter,
        "kan_hidden_dim": args.poster_kan_hidden_dim,
        "kan_num_knots": args.poster_kan_knots,
    }
    if variant == "poster-kan-pa":
        return POSTER_KAN_PA(
            **common_kwargs,
            num_prototypes=args.pa_num_prototypes,
            prototype_scale_init=args.pa_prototype_scale_init,
            use_landmark_gate_fusion=not args.pa_no_gate,
        )
    return POSTER_KAN(**common_kwargs)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return build_model_variant(args.model_variant, args)


def build_criterion(args: argparse.Namespace, class_counts: torch.Tensor | None) -> torch.nn.Module:
    if args.loss == "balanced-softmax":
        if class_counts is None:
            class_counts = torch.ones(7)
        return BalancedSoftmaxLoss(class_counts, label_smoothing=args.label_smoothing)
    return torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def load_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def adapt_state_dict_for_model(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    if not state_dict or not model_state:
        return state_dict

    state_has_module = next(iter(state_dict)).startswith("module.")
    model_has_module = next(iter(model_state)).startswith("module.")
    if state_has_module == model_has_module:
        return state_dict
    if state_has_module and not model_has_module:
        return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    return {f"module.{key}": value for key, value in state_dict.items()}


def load_model_checkpoint(model: torch.nn.Module, checkpoint: dict[str, Any]) -> None:
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = adapt_state_dict_for_model(model, state_dict)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(f"=> loaded {len(compatible)} compatible tensors")


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> SAM:
    if args.optimizer == "adamw":
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == "adam":
        base_optimizer = torch.optim.Adam
    else:
        base_optimizer = torch.optim.SGD

    target = model.module if hasattr(model, "module") else model
    backbone_params = []
    new_module_params = []
    for name, parameter in target.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("landmark_gate_fusion", "kan_head")):
            new_module_params.append(parameter)
        elif args.freeze_backbone:
            parameter.requires_grad = False
        else:
            backbone_params.append(parameter)

    param_groups = []
    backbone_lr = args.lr * args.poster_kan_backbone_lr_scale
    if backbone_params and backbone_lr > 0.0:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if new_module_params:
        param_groups.append({"params": new_module_params, "lr": args.poster_kan_new_lr or args.lr})
    if not param_groups:
        raise ValueError("No trainable parameters left after optimizer grouping.")

    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if base_optimizer is torch.optim.SGD:
        optimizer_kwargs["momentum"] = args.momentum
    return SAM(param_groups, base_optimizer, rho=0.05, adaptive=False, **optimizer_kwargs)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def print_model_profile(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if args.no_profile:
        return
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
        print(f"=> profile params: {params / 1e6:.3f}M")
        print(f"=> profile flops: {flops / 1e9:.3f}G")
    except Exception as exc:
        print(f"=> profile unavailable: {exc}")
    finally:
        model.train(was_training)


def benchmark_model_speed(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if not args.benchmark_speed:
        return
    device = args.device
    sample = torch.randn(1, 3, 224, 224, device=device)
    was_training = model.training
    model.eval()

    def run_once() -> None:
        with torch.no_grad():
            model(sample)

    for _ in range(3):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timer = benchmark.Timer(stmt="run_once()", globals={"run_once": run_once})
    result = timer.timeit(20)
    print(f"=> benchmark median: {result.median * 1000:.3f} ms")
    model.train(was_training)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        result = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    teacher_model: torch.nn.Module | None,
    criterion: torch.nn.Module,
    optimizer: SAM,
    epoch: int,
    args: argparse.Namespace,
) -> tuple[float, float]:
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(len(train_loader), [losses, top1], prefix=f"Epoch: [{epoch}]")
    model.train()

    for index, (images, target) in enumerate(train_loader):
        images = images.to(args.device, non_blocking=args.device.type == "cuda")
        target = target.to(args.device, non_blocking=args.device.type == "cuda")
        teacher_logits = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(images)

        output = model(images)
        loss = criterion(output, target)
        if teacher_logits is not None:
            loss = loss + args.distill_lambda * distillation_loss(output, teacher_logits, args.distill_temperature)

        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss, images.size(0))
        top1.update(acc1, images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        output = model(images)
        loss = criterion(output, target)
        if teacher_logits is not None:
            loss = loss + args.distill_lambda * distillation_loss(output, teacher_logits, args.distill_temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.second_step(zero_grad=True)

        if index % args.print_freq == 0:
            progress.display(index)

    return top1.avg, losses.avg


def validate(
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[float, float, np.ndarray]:
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(len(val_loader), [losses, top1], prefix="Test: ")
    confusion = np.zeros((7, 7), dtype=np.int64)
    model.eval()
    criterion.eval()

    with torch.no_grad():
        for index, (images, target) in enumerate(val_loader):
            images = images.to(args.device, non_blocking=args.device.type == "cuda")
            target = target.to(args.device, non_blocking=args.device.type == "cuda")
            output = model(images)
            loss = criterion(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss, images.size(0))
            top1.update(acc1, images.size(0))
            pred = output.argmax(dim=1).detach().cpu().numpy()
            truth = target.detach().cpu().numpy()
            confusion += metrics.confusion_matrix(truth, pred, labels=list(range(7)))

            if index % args.print_freq == 0:
                progress.display(index)

    print(f" **** Accuracy {top1.avg:.3f} *** ")
    print(confusion)
    return top1.avg, losses.avg, confusion


def save_checkpoint(state: dict[str, Any], is_best: bool, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "last.pth")
    if is_best:
        best_state = {key: value for key, value in state.items() if key != "optimizer"}
        torch.save(best_state, output_dir / "best.pth")


def main() -> None:
    args = parse_args()
    if args.resume and args.init_weights:
        raise ValueError("--resume and --init-weights are mutually exclusive")

    args.device = resolve_device(args)
    args.output_dir = Path(args.output_dir)
    print("Run started:", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Device:", args.device)

    model = build_model(args)
    print_model_profile(model, args)
    if args.device.type == "cuda":
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        model = model.to(args.device)
    benchmark_model_speed(model, args)

    teacher_model = None
    if args.teacher_checkpoint and args.distill_lambda > 0:
        teacher_model = build_model_variant(args.teacher_model_variant, args)
        teacher_model = torch.nn.DataParallel(teacher_model).to(args.device) if args.device.type == "cuda" else teacher_model.to(args.device)
        load_model_checkpoint(teacher_model, load_checkpoint(args.teacher_checkpoint))
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_acc = torch.tensor(0.0)

    if args.init_weights:
        print(f"=> initializing from {args.init_weights}")
        load_model_checkpoint(model, load_checkpoint(args.init_weights))

    if args.resume:
        print(f"=> resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        args.start_epoch = checkpoint.get("epoch", args.start_epoch)
        best_acc = torch.as_tensor(checkpoint.get("best_acc", best_acc))
        load_model_checkpoint(model, checkpoint)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state(optimizer, args.device)

    cudnn.benchmark = args.device.type == "cuda"

    train_dataset = None
    class_counts = None
    if args.evaluate is None:
        train_dataset = datasets.ImageFolder(os.path.join(args.data, "train"), build_train_transform(args.data_type))
        class_counts = compute_class_counts(train_dataset, num_classes=7)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.device.type == "cuda",
        )

    criterion = build_criterion(args, class_counts).to(args.device)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, "valid"), build_eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
    )

    if args.evaluate:
        print(f"=> evaluating {args.evaluate}")
        load_model_checkpoint(model, load_checkpoint(args.evaluate))
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("Current param group lrs:", [group["lr"] for group in optimizer.param_groups])
        train_acc, train_loss = train_one_epoch(train_loader, model, teacher_model, criterion, optimizer, epoch, args)
        val_acc, val_loss, confusion = validate(val_loader, model, criterion, args)
        scheduler.step()

        val_acc_tensor = torch.as_tensor(val_acc)
        is_best = bool(val_acc_tensor > best_acc)
        best_acc = torch.maximum(best_acc, val_acc_tensor)
        print(f"Epoch {epoch}: train_acc={train_acc:.3f} train_loss={train_loss:.4f} val_acc={val_acc:.3f} val_loss={val_loss:.4f}")
        print("Current best accuracy:", float(best_acc.item()))

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "confusion": confusion,
            },
            is_best=is_best,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
