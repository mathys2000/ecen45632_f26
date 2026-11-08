"""
audio_cnn.py -- Week 14, Lecture 28
AudioCNN + training loop + SpecAugment, with the lecture's checks built in.

Run:
    python audio_cnn.py                 # checks, then train/val/test on synthetic log-Mel data
    python audio_cnn.py --checks-only   # just the checks
    python audio_cnn.py --no-specaug --dropout 0.0   # baseline for the 3-run experiment (HW 5)

The data are synthetic [1, M, T] log-Mel patches (ridge / onset / chirp / texture) with a
per-source "coloration", so the whole pipeline runs with no audio files. To use real audio,
replace make_synthetic() with your Lecture 27 Dataset; everything else is unchanged.
"""
from __future__ import annotations

import argparse
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------
# Reproducibility (slide 27 of Lecture 27)
# --------------------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy inside DataLoader workers (torch is already seeded per worker)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------------------
# Model (slides 4-13)
# --------------------------------------------------------------------------------------


class Block(nn.Module):
    """Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2.  Conv has no bias because BN follows it."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AudioCNN(nn.Module):
    """Input [B, 1, M, T] log-Mel -> logits [B, n_classes].  T may vary (needs T >= 8)."""

    def __init__(self, n_classes: int, p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(Block(1, 16), Block(16, 32), Block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input contract: fail loudly at the boundary instead of deep inside Conv2d.
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"expected [B, 1, M, T], got {tuple(x.shape)} "
                "(did an unconstrained squeeze() remove the channel axis?)"
            )
        if x.shape[-1] < 8:
            raise ValueError(f"T={x.shape[-1]} is too short for three 2x2 pools (need T >= 8)")
        return self.head(self.pool(self.features(x)))  # raw logits, NO softmax


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------------------
# SpecAugment (slides 31-35): training only, applied AFTER normalization so 0 == mean level
# --------------------------------------------------------------------------------------


def mask_axis(x: torch.Tensor, axis: int, max_width: int) -> torch.Tensor:
    """Zero a random contiguous band of width in [0, max_width] along `axis`."""
    width = int(torch.randint(0, max_width + 1, ()).item())
    width = min(width, x.shape[axis])
    if width == 0:
        return x
    start = int(torch.randint(0, x.shape[axis] - width + 1, ()).item())
    index = [slice(None)] * x.ndim
    index[axis] = slice(start, start + width)
    x = x.clone()  # never modify the cached/source tensor in place
    x[tuple(index)] = 0.0
    return x


class SpecAugment:
    def __init__(self, max_freq_width: int = 8, max_time_width: int = 16):
        self.max_freq_width = max_freq_width
        self.max_time_width = max_time_width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # x: [1, M, T]
        x = mask_axis(x, -2, self.max_freq_width)  # frequency (Mel) mask
        x = mask_axis(x, -1, self.max_time_width)  # time mask
        return x


def mask_duration_seconds(width_frames: int, hop_length: int, sample_rate: int) -> float:
    """Wt * hop / fs  (slide 34)."""
    return width_frames * hop_length / sample_rate


# --------------------------------------------------------------------------------------
# Data: synthetic log-Mel patches, group-aware splitting, Dataset
# --------------------------------------------------------------------------------------

CLASS_NAMES = ["tone", "onset", "chirp", "texture"]


def make_synthetic(n_per_group: int = 8, n_groups: int = 30, M: int = 64, T: int = 173,
                   seed: int = 0):
    """Return X [N,1,M,T] float, y [N] long, g [N] long (source/group id).

    Each group (a 'source': speaker/device/room) has its own fixed per-Mel-bin coloration, so
    a model that memorizes source identity would look good on a clip-level split and fail on
    a group-level split -- exactly the leakage the lecture warns about.
    """
    gen = torch.Generator().manual_seed(seed)
    n_classes = len(CLASS_NAMES)
    X, Y, G = [], [], []
    for g in range(n_groups):
        color = 0.6 * torch.randn(M, 1, generator=gen)  # per-source spectral tilt/coloration
        for i in range(n_per_group):
            y = (g * n_per_group + i) % n_classes
            x = torch.randn(1, M, T, generator=gen) + color
            if y == 0:  # horizontal ridge: sustained tone
                r = int(torch.randint(8, M - 8, (1,), generator=gen))
                t0 = int(torch.randint(0, T // 3, (1,), generator=gen))
                x[0, r - 1:r + 2, t0:t0 + T // 2] += 3.0
            elif y == 1:  # vertical stripe: onset / click
                t0 = int(torch.randint(10, T - 10, (1,), generator=gen))
                x[0, :, t0:t0 + 3] += 3.0
            elif y == 2:  # diagonal ridge: chirp
                sign = 1 if torch.rand(1, generator=gen).item() < 0.5 else -1
                r0 = int(torch.randint(16, M - 16, (1,), generator=gen))
                t0 = int(torch.randint(0, T - 70, (1,), generator=gen))
                for k in range(60):
                    r = min(max(r0 + sign * (k // 2), 1), M - 2)
                    x[0, r - 1:r + 2, t0 + k] += 3.0
            else:  # broadband texture: extra noise energy
                x += 1.2 * torch.randn(1, M, T, generator=gen)
            X.append(x)
            Y.append(y)
            G.append(g)
    return torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)


def group_split(groups: torch.Tensor, fractions=(0.6, 0.2, 0.2), seed: int = 0):
    """Assign WHOLE groups to train/val/test, then return sample indices for each split."""
    uniq = sorted(set(groups.tolist()))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    split_groups = [set(uniq[:n_train]), set(uniq[n_train:n_train + n_val]),
                    set(uniq[n_train + n_val:])]
    return [[i for i, g in enumerate(groups.tolist()) if g in sg] for sg in split_groups]


class MelDataset(Dataset):
    """Returns (features [1,M,T] normalized, label long scalar, group id).

    Normalization statistics are passed in (fit on TRAIN only). `transform` is SpecAugment for
    training and None for validation/test, so evaluation inputs are deterministic.
    """

    def __init__(self, X, y, g, indices, mean: float, std: float, transform=None):
        self.X, self.y, self.g = X, y, g
        self.indices = list(indices)
        self.mean, self.std = float(mean), max(float(std), 1e-6)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        x = (self.X[j] - self.mean) / self.std  # [1, M, T]
        if self.transform is not None:
            x = self.transform(x)  # masks come AFTER normalization
        return x, self.y[j], self.g[j]


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      num_workers=num_workers, worker_init_fn=seed_worker, drop_last=False)


# --------------------------------------------------------------------------------------
# Training / validation epoch (slides 16-19, 23)
# --------------------------------------------------------------------------------------


def _check_batch(logits: torch.Tensor, y: torch.Tensor) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, C], got {tuple(logits.shape)}")
    if y.dtype != torch.long or y.ndim != 1 or y.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must be long [B]; got {y.dtype} {tuple(y.shape)}")
    if int(y.min()) < 0 or int(y.max()) >= logits.shape[1]:
        raise ValueError("target index outside [0, n_classes)")


def run_epoch(model, loader, loss_fn, device, optimizer=None, n_classes=None):
    """One pass over `loader`. Trains if `optimizer` is given, otherwise evaluates.

    Returns dict(loss, acc, bal_acc, recall, n). Loss is the example-weighted mean
    (sum of per-batch mean * batch size / N), not a mean of batch means.
    """
    training = optimizer is not None
    model.train(training)  # dropout + BatchNorm behave differently
    context = torch.enable_grad() if training else torch.inference_mode()
    total_loss, total_correct, n = 0.0, 0, 0
    confusion = None
    with context:
        for x, y, _g in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)  # gradients accumulate by default
            logits = model(x)
            _check_batch(logits, y)
            loss = loss_fn(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            bs = y.shape[0]
            total_loss += loss.item() * bs
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            n += bs
            if confusion is None:
                C = n_classes or logits.shape[1]
                confusion = torch.zeros(C, C, dtype=torch.long)
            idx = y.cpu() * confusion.shape[1] + pred.cpu()
            confusion += torch.bincount(idx, minlength=confusion.numel()).view_as(confusion)
    per_class = confusion.diag().float() / confusion.sum(1).clamp_min(1).float()
    present = confusion.sum(1) > 0
    return {
        "loss": total_loss / n,
        "acc": total_correct / n,
        "bal_acc": per_class[present].mean().item(),  # class-sensitive metric
        "recall": per_class.tolist(),
        "n": n,
    }


def diagnose(history: dict, n_classes: int) -> str:
    """Name the learning-curve pattern (slide 26). Heuristic, not a substitute for looking."""
    tr, va = history["train_loss"], history["val_loss"]
    best = int(np.argmin(va))
    chance = math.log(n_classes)
    if va[-1] > 1.1 * va[best] and tr[-1] < tr[0]:
        return (f"OVERFIT: val loss bottomed at epoch {best + 1} ({va[best]:.3f}) then rose to "
                f"{va[-1]:.3f} while train loss fell. Try dropout, SpecAugment, weight decay, "
                "a smaller model, more data, or early stopping.")
    if tr[-1] > 0.7 * chance:
        return ("UNDERFIT: train loss is still near chance "
                f"({tr[-1]:.3f} vs ln(C)={chance:.3f}). Verify targets/LR first, then capacity.")
    return f"HEALTHY: train {tr[-1]:.3f}, val {va[-1]:.3f}, gap {va[-1] - tr[-1]:+.3f}."


def fit(model, train_loader, val_loader, device, n_classes, epochs=30, lr=3e-3,
        weight_decay=1e-3, patience=8, verbose=True):
    """AdamW + cosine annealing (stepped once per epoch). Checkpoint = best VAL loss."""
    loss_fn = nn.CrossEntropyLoss()  # raw logits + long targets
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
            "val_bal_acc": [], "lr": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for epoch in range(1, epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer, n_classes)
        va = run_epoch(model, val_loader, loss_fn, device, None, n_classes)
        scheduler.step()  # once per epoch because T_max is in epochs
        for k, v in (("train_loss", tr["loss"]), ("val_loss", va["loss"]),
                     ("train_acc", tr["acc"]), ("val_acc", va["acc"]),
                     ("val_bal_acc", va["bal_acc"]), ("lr", cur_lr)):
            hist[k].append(v)
        if verbose:
            print(f"epoch {epoch:3d} | lr {cur_lr:.2e} | train {tr['loss']:.3f}/{tr['acc']:.3f}"
                  f" | val {va['loss']:.3f}/{va['acc']:.3f} (bal {va['bal_acc']:.3f})")
        if va["loss"] < best_val - 1e-4:  # declared selection criterion: val loss
            best_val, best_epoch, bad = va["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break
    model.load_state_dict(best_state)
    return hist, best_epoch


# --------------------------------------------------------------------------------------
# Checks (slides 7, 9, 13, 19, 23, 24, 34, 36)
# --------------------------------------------------------------------------------------


def audit_shapes(model: AudioCNN, B: int = 8, M: int = 64, T: int = 173, verbose: bool = True):
    """Forward hooks print the real shape leaving each stage (activity step 2)."""
    rows, hooks = [], []

    def hook(name):
        def _h(_m, _inp, out):
            rows.append((name, tuple(out.shape)))
        return _h

    for i, blk in enumerate(model.features):
        hooks.append(blk.body[0].register_forward_hook(hook(f"block{i + 1}.conv")))
        hooks.append(blk.register_forward_hook(hook(f"block{i + 1}.pool")))
    hooks.append(model.pool.register_forward_hook(hook("adaptive_pool")))
    hooks.append(model.head.register_forward_hook(hook("logits")))
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(B, 1, M, T))
    model.train(was_training)
    for h in hooks:
        h.remove()
    if verbose:
        print(f"  input            {(B, 1, M, T)}")
        for name, shape in rows:
            print(f"  {name:<16} {shape}")
    return rows


def check_shape_contract(n_classes: int) -> None:
    model = AudioCNN(n_classes)
    for T, pooled_w in ((128, 16), (173, 21)):
        rows = dict(audit_shapes(model, B=4, T=T, verbose=False))
        assert rows["block3.pool"] == (4, 64, 8, pooled_w), rows["block3.pool"]
        assert rows["adaptive_pool"] == (4, 64, 1, 1)
        assert rows["logits"] == (4, n_classes)
    # the input contract must reject a squeezed tensor
    try:
        model(torch.zeros(4, 64, 173))
    except ValueError:
        pass
    else:
        raise AssertionError("model accepted a 3-D input")
    print("  ok: T=128 -> 8x16, T=173 -> 8x21, both -> [B,64,1,1] -> [B,C]; 3-D input rejected")


def check_param_counts(n_classes: int = 10) -> None:
    assert count_params(Block(16, 32)) == 4608 + 64  # conv (no bias) + BN scale/shift
    head = nn.Linear(64, n_classes)
    assert count_params(head) == 64 * n_classes + n_classes  # 650 for C=10
    assert 64 * 8 * 16 * n_classes + n_classes == 81_930 or n_classes != 10
    total = count_params(AudioCNN(n_classes))
    print(f"  ok: Conv16->32+BN = 4,672; head(64->{n_classes}) = {count_params(head)}; "
          f"total = {total:,}")


def check_train_eval_modes(n_classes: int) -> None:
    model = AudioCNN(n_classes, p=0.5)
    x = torch.randn(8, 1, 64, 173)
    model.eval()
    with torch.inference_mode():
        a, b = model(x), model(x)
    assert torch.allclose(a, b), "eval mode must be deterministic"
    probe = copy.deepcopy(model)  # train-mode forward updates BN stats; don't touch the original
    probe.train()
    with torch.no_grad():
        c, d = probe(x), probe(x)
    assert not torch.allclose(c, d), "dropout should make train-mode outputs differ"
    print("  ok: eval() is deterministic; train() with dropout is stochastic")


def check_group_split(groups: torch.Tensor, splits) -> None:
    sets = [set(groups[idx].tolist()) for idx in splits]
    for i in range(3):
        for j in range(i + 1, 3):
            assert sets[i].isdisjoint(sets[j]), f"group leakage between splits {i} and {j}"
    assert sum(len(s) for s in splits) == len(groups), "samples lost or duplicated"
    print(f"  ok: train/val/test groups are disjoint "
          f"({len(sets[0])}/{len(sets[1])}/{len(sets[2])} groups)")


def check_val_deterministic(X, y, g, val_idx, mean, std) -> None:
    ds = MelDataset(X, y, g, val_idx, mean, std, transform=None)
    a, _, _ = ds[0]
    b, _, _ = ds[0]
    assert torch.equal(a, b), "validation transform must be deterministic"
    aug = MelDataset(X, y, g, val_idx, mean, std, transform=SpecAugment(8, 16))
    torch.manual_seed(1)
    outs = [aug[0][0] for _ in range(10)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), "SpecAugment never changed input"
    assert not torch.equal(X[val_idx[0]], X[val_idx[0]] * 0), "sanity"
    print("  ok: validation items are identical across calls; SpecAugment varies them")


def check_mask_duration() -> None:
    assert abs(mask_duration_seconds(20, 160, 16000) - 0.20) < 1e-12
    assert abs(mask_duration_seconds(24, 160, 16000) - 0.24) < 1e-12
    assert abs(mask_duration_seconds(10, 160, 16000) - 0.10) < 1e-12
    print("  ok: 20 frames = 0.20 s, 24 = 0.24 s, 10 = 0.10 s at hop 160 / 16 kHz")


def check_scheduler() -> None:
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "cosine LR should not increase"
    assert opt.param_groups[0]["lr"] < 1e-8 + 1e-3 * lrs[0], "LR should approach eta_min"
    print(f"  ok: LR {lrs[0]:.1e} -> {opt.param_groups[0]['lr']:.1e} over 10 epoch-steps")


def check_overfit_one_batch(dataset: Dataset, n_classes: int, device) -> None:
    """If the model can't memorize ONE batch with regularization off, something is broken."""
    xs, ys = [], []
    for i in range(min(16, len(dataset))):
        x, y, _ = dataset[i]
        xs.append(x)
        ys.append(y)
    x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
    model = AudioCNN(n_classes, p=0.0).to(device)  # dropout off, no SpecAugment
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    first = None
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.1, f"could not overfit one batch (loss {first:.3f} -> {loss.item():.3f})"
    print(f"  ok: one-batch overfit, loss {first:.3f} -> {loss.item():.3f}")


def run_checks(X, y, g, splits, mean, std, device) -> None:
    C = len(CLASS_NAMES)
    print("Checks")
    check_shape_contract(C)
    check_param_counts(10)
    check_train_eval_modes(C)
    check_group_split(g, splits)
    check_val_deterministic(X, y, g, splits[1], mean, std)
    check_mask_duration()
    check_scheduler()
    train_plain = MelDataset(X, y, g, splits[0], mean, std, transform=None)
    check_overfit_one_batch(train_plain, C, device)
    print("All checks passed.\n")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-specaug", action="store_true")
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    C = len(CLASS_NAMES)

    X, y, g = make_synthetic(seed=args.seed)
    splits = group_split(g, seed=args.seed)  # whole groups -> splits BEFORE any statistics
    train_idx, val_idx, test_idx = splits
    mean = X[train_idx].mean().item()  # normalization stats: TRAIN only
    std = X[train_idx].std().item()

    run_checks(X, y, g, splits, mean, std, device)
    if args.checks_only:
        return

    print("Shape audit (B=8, T=173):")
    audit_shapes(AudioCNN(C, args.dropout), B=8, T=173)
    print()

    specaug = None if args.no_specaug else SpecAugment(max_freq_width=8, max_time_width=16)
    if specaug:
        print(f"Max time mask = {mask_duration_seconds(16, 160, 16000):.2f} s at hop 160 / 16 kHz")
    train_ds = MelDataset(X, y, g, train_idx, mean, std, transform=specaug)
    val_ds = MelDataset(X, y, g, val_idx, mean, std, transform=None)  # deterministic, unmasked
    test_ds = MelDataset(X, y, g, test_idx, mean, std, transform=None)
    train_loader = make_loader(train_ds, args.batch_size, True, args.seed)
    val_loader = make_loader(val_ds, args.batch_size, False, args.seed)
    test_loader = make_loader(test_ds, args.batch_size, False, args.seed)

    # Inspect the first batch (shape, range, class counts, source ids)
    xb, yb, gb = next(iter(train_loader))
    print(f"First batch: x {tuple(xb.shape)} range [{xb.min():.2f}, {xb.max():.2f}], "
          f"classes {torch.bincount(yb, minlength=C).tolist()}, groups {sorted(set(gb.tolist()))}\n")

    model = AudioCNN(C, p=args.dropout).to(device)
    print(f"Trainable parameters: {count_params(model):,}\n")
    hist, best_epoch = fit(model, train_loader, val_loader, device, C, epochs=args.epochs,
                           lr=args.lr, weight_decay=args.weight_decay)
    print("\n" + diagnose(hist, C))

    # Test set is touched exactly once, after the checkpoint (best val loss) is fixed.
    te = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device, None, C)
    print(f"\nBest epoch {best_epoch} | TEST loss {te['loss']:.3f} acc {te['acc']:.3f} "
          f"balanced acc {te['bal_acc']:.3f}")
    print("Per-class recall:", {n: round(r, 3) for n, r in zip(CLASS_NAMES, te["recall"])})
    print("Config:", vars(args))


if __name__ == "__main__":
    main()