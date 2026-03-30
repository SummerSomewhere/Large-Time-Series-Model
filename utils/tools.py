import os
import sys
import tempfile

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

plt.switch_backend('agg')


def _atomic_torch_save(obj, final_path: str) -> None:
    """Write to a temp file in the same directory, then os.replace (avoids truncated files if process dies mid-write)."""
    final_path = os.path.abspath(final_path)
    directory = os.path.dirname(final_path)
    os.makedirs(directory, exist_ok=True)
    _root, ext = os.path.splitext(final_path)
    tmp_suffix = (ext + ".tmp") if ext else ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=tmp_suffix, prefix="ckpt_", dir=directory)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, final_path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class LargeScheduler:
    def __init__(self, args, optimizer) -> None:
        super().__init__()
        self.learning_rate = args.learning_rate
        self.decay_fac = args.decay_fac
        self.lradj = args.lradj
        self.use_multi_gpu = args.use_multi_gpu
        self.optimizer = optimizer
        self.args = args
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def schedule_epoch(self, epoch: int):
        if self.lradj == 'type1':
            lr_adjust = {epoch: self.learning_rate if epoch < 3 else self.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif self.lradj == 'type2':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch - 1) // 1))}
        elif self.lradj == 'type4':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch) // 1))}
        elif self.lradj == 'type3':
            self.learning_rate = 1e-4
            lr_adjust = {epoch: self.learning_rate if epoch < 3 else self.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif self.lradj == 'cos_epoch':
            lr_adjust = {epoch: self.learning_rate / 2 * (1 + math.cos(epoch / self.args.cos_max_decay_epoch * math.pi))}
        else:
            return

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                tag = param_group.get("finetune_group", "main")
                if tag == "main":
                    param_group["lr"] = lr
                elif tag == "res_omega":
                    param_group["lr"] = float(
                        getattr(self.args, "finetune_res_omega_lr", 1e-5)
                    )
                elif tag == "res_lambda":
                    param_group["lr"] = float(
                        getattr(self.args, "finetune_res_lambda_lr", 1e-5)
                    )
                else:
                    param_group["lr"] = lr
            print('Updating learning rate to {}'.format(lr))

    def schedule_step(self, n: int):
        if self.lradj == 'cos_step':
            if n < self.args.cos_warm_up_steps:
                res = (self.args.cos_max - self.learning_rate) / self.args.cos_warm_up_steps * n + self.learning_rate
                self.last = res
            else:
                t = (n - self.args.cos_warm_up_steps) / (self.args.cos_max_decay_steps - self.args.cos_warm_up_steps)
                t = min(t, 1.0)
                res = self.args.cos_min + 0.5 * (self.args.cos_max - self.args.cos_min) * (1 + np.cos(t * np.pi))
                self.last = res
        else:
            return

        for param_group in self.optimizer.param_groups:
            tag = param_group.get("finetune_group", "main")
            if tag == "main":
                param_group["lr"] = res
            elif tag == "res_omega":
                param_group["lr"] = float(
                    getattr(self.args, "finetune_res_omega_lr", 1e-5)
                )
            elif tag == "res_lambda":
                param_group["lr"] = float(
                    getattr(self.args, "finetune_res_lambda_lr", 1e-5)
                )
            else:
                param_group["lr"] = res
        if n % 500 == 0:
            print('Updating learning rate to {}'.format(res))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, local_rank=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # Under torchrun DDP, only rank 0 should write checkpoints (avoids corrupt / racing writes).
        self.local_rank = local_rank

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # Lightning-style .ckpt: top-level "state_dict" (Timer/TrmEncoder load with .ckpt + from_lightning_ckpt=True).
        def _write():
            if self.verbose:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
                )
            sd = model.state_dict()
            _atomic_torch_save(sd, os.path.join(path, 'checkpoint.pth'))
            _atomic_torch_save({"state_dict": sd}, os.path.join(path, 'checkpoint.ckpt'))

        if dist.is_initialized():
            if self.local_rank == 0:
                _write()
            dist.barrier()
        else:
            _write()
        self.val_loss_min = val_loss


class EarlyStoppingLarge:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.use_multi_gpu = args.use_multi_gpu
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            # self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            # self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0
        if self.use_multi_gpu:
            if self.local_rank == 0:
                self.save_checkpoint(val_loss, model, path, epoch)
            dist.barrier()
        else:
            self.save_checkpoint(val_loss, model, path, epoch)
        return self.best_epoch

    def save_checkpoint(self, val_loss, model, path, epoch):
        sd = model.state_dict()
        _atomic_torch_save(sd, os.path.join(path, f'checkpoint_{epoch}.pth'))
        _atomic_torch_save({"state_dict": sd}, os.path.join(path, f'checkpoint_{epoch}.ckpt'))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', c='dodgerblue', linewidth=2)
    plt.plot(true, label='GroundTruth', c='tomato', linewidth=2)
    plt.legend(loc='upper left')
    plt.savefig(name, bbox_inches='tight')


def attn_map(attn, path='./pic/attn_map.pdf'):
    """
    Attention map visualization
    """
    plt.figure()
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def fft_mae_time_series(
    pred: torch.Tensor,
    true: torch.Tensor,
    time_dim: int = 1,
    normalize_energy: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MAE between normalized magnitude spectra (energy envelope) along time_dim.
    pred/true same shape; encourages matching dominant frequency content.
    """
    if pred.shape != true.shape:
        raise ValueError(f"fft_mae_time_series: shape mismatch {pred.shape} vs {true.shape}")
    fp = torch.fft.rfft(pred, dim=time_dim)
    ft = torch.fft.rfft(true, dim=time_dim)
    ep = torch.abs(fp)
    et = torch.abs(ft)
    if normalize_energy:
        ep = ep / (ep.sum(dim=time_dim, keepdim=True) + eps)
        et = et / (et.sum(dim=time_dim, keepdim=True) + eps)
    return (ep - et).abs().mean()


def fft_complex_mae(pred: torch.Tensor, true: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
    """
    Mean absolute magnitude of complex rFFT coefficient differences (TimeEmb-style spectral term).
    """
    if pred.shape != true.shape:
        raise ValueError(f"fft_complex_mae: shape mismatch {pred.shape} vs {true.shape}")
    fp = torch.fft.rfft(pred, dim=time_dim)
    ft = torch.fft.rfft(true, dim=time_dim)
    return (fp - ft).abs().mean()


def infer_resonance_period_hours_from_freq(freq_str) -> float:
    """
    Nominal cycle length in hours for ω = 1/period when physical timestamps use dt in hours (e.g. ETTh1 hourly → 24h day).
    """
    if freq_str is None or str(freq_str).strip() == "":
        return 24.0
    s = str(freq_str).strip().lower()
    if s == "h" or s.startswith("h"):
        return 24.0
    if s == "d" or s == "b" or s.startswith("d"):
        return 168.0
    if "min" in s:
        return 24.0
    return 24.0


def build_timer_finetune_param_groups(model, args):
    """
    Split trainable Timer params: res_omega / res_lambda use finetune_res_*_lr; others use args.learning_rate.
    Each group may include 'finetune_group' for LargeScheduler (main vs resonance lrs).
    """
    m = model.module if hasattr(model, "module") else model
    omega_p, lambda_p, other_p = [], [], []
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("res_omega") or "har_omega_raw" in name:
            omega_p.append(p)
        elif name.endswith("res_lambda"):
            lambda_p.append(p)
        else:
            other_p.append(p)
    wd = float(args.weight_decay) if int(getattr(args, "use_weight_decay", 0)) else 0.0
    lr_main = float(args.learning_rate)
    lr_o = float(getattr(args, "finetune_res_omega_lr", 1e-5))
    lr_l = float(getattr(args, "finetune_res_lambda_lr", 1e-5))
    groups = []
    if other_p:
        groups.append(
            {
                "params": other_p,
                "lr": lr_main,
                "weight_decay": wd,
                "finetune_group": "main",
            }
        )
    if omega_p:
        groups.append(
            {
                "params": omega_p,
                "lr": lr_o,
                "weight_decay": wd,
                "finetune_group": "res_omega",
            }
        )
    if lambda_p:
        groups.append(
            {
                "params": lambda_p,
                "lr": lr_l,
                "weight_decay": wd,
                "finetune_group": "res_lambda",
            }
        )
    return groups


def apply_timer_finetune_freeze(model, args) -> None:
    """
    Timer only: freeze most weights for partial finetune. Unwrap DDP via .module when present.
    Modes (args.finetune_trainable):
      - full: no-op
      - last_layer: last EncoderLayer FFN + norms + inner_attention + proj; Q/K/V/out Linear in
        AttentionLayer stay frozen (no query/key/value/out_projection grads).
      - resonance_only: freeze entire backbone; only train res_omega, res_lambda, res_phi on last inner_attention.
      - last_attention_only: freeze entire backbone except last EncoderLayer.attention (Q/K/V/out + resonance inner).
      - harmonic_proj: train only harmonic gated inner_attention + backbone.proj (needs harmonic_gated_resonance=1).
    """
    mode = getattr(args, "finetune_trainable", "full")
    if mode is None or mode == "full":
        return
    if getattr(args, "model", None) != "Timer":
        return

    m = model.module if hasattr(model, "module") else model

    for p in m.parameters():
        p.requires_grad = False

    if mode == "last_layer":
        last = m.backbone.decoder.attn_layers[-1]
        attn = last.attention
        for mod in (last.norm1, last.norm2, last.conv1, last.conv2, attn.inner_attention):
            for p in mod.parameters():
                p.requires_grad = True
        for p in m.backbone.proj.parameters():
            p.requires_grad = True
        # Frozen: attn.query_projection, key_projection, value_projection, out_projection
    elif mode == "resonance_only":
        if not int(getattr(args, "resonance_last_layer", 0)):
            raise ValueError("finetune_trainable=resonance_only requires resonance_last_layer=1")
        inner = m.backbone.decoder.attn_layers[-1].attention.inner_attention
        n = 0
        for name, p in inner.named_parameters():
            if name.startswith("res_"):
                p.requires_grad = True
                n += p.numel()
        if n == 0:
            raise RuntimeError("No res_* trainable params; is last layer FullAttentionLastLayerResonance?")
    elif mode == "last_attention_only":
        if not int(getattr(args, "resonance_last_layer", 0)):
            raise ValueError("finetune_trainable=last_attention_only requires resonance_last_layer=1")
        last = m.backbone.decoder.attn_layers[-1]
        for p in last.attention.parameters():
            p.requires_grad = True
    elif mode == "harmonic_proj":
        if not int(getattr(args, "harmonic_gated_resonance", 0)):
            raise ValueError("finetune_trainable=harmonic_proj requires harmonic_gated_resonance=1")
        inner = m.backbone.decoder.attn_layers[-1].attention.inner_attention
        for p in inner.parameters():
            p.requires_grad = True
        for p in m.backbone.proj.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown finetune_trainable={mode!r}")


class HiddenPrints:
    def __init__(self, rank):
        if rank is None:
            rank = 0
        self.rank = rank
    def __enter__(self):
        if self.rank == 0:
            return
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank == 0:
            return
        sys.stdout.close()
        sys.stdout = self._original_stdout