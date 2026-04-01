import os
import sys
import tempfile

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

plt.switch_backend('agg')

# Timer SIG-Gate: alpha-only param group uses this multiplier vs base lr (weight_decay=0).
SIG_GATE_ALPHA_LR_MULT = 100.0


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


class BetaScheduler:
    """
    VIB KL weight: beta=0 for the first warmup_ratio * total_steps steps, then linear ramp
    to beta_max so the last training step (index total_steps - 1) reaches beta_max.
    """

    def __init__(self, total_steps: int, warmup_ratio: float = 0.1, beta_max: float = 1e-4):
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(round(float(warmup_ratio) * self.total_steps)))
        self.beta_max = float(beta_max)

    def __call__(self, step: int) -> float:
        step = max(0, int(step))
        if step < self.warmup_steps:
            return 0.0
        ramp_end = self.total_steps - 1
        if ramp_end <= self.warmup_steps:
            return self.beta_max
        denom = max(1, ramp_end - self.warmup_steps)
        t = min(1.0, (step - self.warmup_steps) / float(denom))
        return t * self.beta_max


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
                if tag == "sig_gate_alpha":
                    param_group["lr"] = lr * SIG_GATE_ALPHA_LR_MULT
                else:
                    param_group["lr"] = lr
            print(
                'Updating learning rate to {} (gate alpha scalars use {:.0f}x base when present)'.format(
                    lr, SIG_GATE_ALPHA_LR_MULT
                )
            )

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


def build_timer_finetune_param_groups(model, args):
    """
    Timer: main group (lr + optional weight_decay); optional zero-init gate scalars (legacy sig_gate alpha,
    HM-ISR entry/exit alpha, CrossAttentionBridge alpha) in a high-lr group
    (SIG_GATE_ALPHA_LR_MULT x lr, weight_decay=0).
    """
    m = model.module if hasattr(model, "module") else model
    wd = float(args.weight_decay) if int(getattr(args, "use_weight_decay", 0)) else 0.0
    lr_main = float(args.learning_rate)

    sig_gate = getattr(m, "sig_gate", None)
    sig_alpha_param = sig_gate.alpha if sig_gate is not None else None
    hm_e = getattr(m, "hm_isr_entry", None)
    hm_e_alpha = hm_e.alpha if hm_e is not None else None
    hm_x = getattr(m, "hm_isr_exit", None)
    hm_x_alpha = hm_x.alpha if hm_x is not None else None
    cab = getattr(m, "cross_attn_bridge", None)
    cab_alpha_param = cab.alpha if cab is not None else None

    main_params: list = []
    alpha_params: list = []
    for p in m.parameters():
        if not p.requires_grad:
            continue
        if sig_alpha_param is not None and p is sig_alpha_param:
            alpha_params.append(p)
        elif hm_e_alpha is not None and p is hm_e_alpha:
            alpha_params.append(p)
        elif hm_x_alpha is not None and p is hm_x_alpha:
            alpha_params.append(p)
        elif cab_alpha_param is not None and p is cab_alpha_param:
            alpha_params.append(p)
        else:
            main_params.append(p)

    if not main_params and not alpha_params:
        return []

    groups: list = []
    if main_params:
        groups.append(
            {
                "params": main_params,
                "lr": lr_main,
                "weight_decay": wd,
                "finetune_group": "main",
            }
        )
    if alpha_params:
        groups.append(
            {
                "params": alpha_params,
                "lr": lr_main * SIG_GATE_ALPHA_LR_MULT,
                "weight_decay": 0.0,
                "finetune_group": "sig_gate_alpha",
            }
        )
    return groups


def apply_timer_finetune_freeze(model, args) -> None:
    """
    Timer only: freeze most weights for partial finetune. Unwrap DDP via .module when present.
    Modes (args.finetune_trainable):
      - full: no-op
      - last_layer: last EncoderLayer FFN + norms + inner_attention + proj; Q/K/V/out stay frozen.
      - periodic_emb_proj: freeze backbone; train hour_embed, day_embed, periodic_gamma, proj
        (requires periodic_embedding_branch=1).
      - proj_only: freeze backbone; train Linear patch head (proj) only — fair baseline vs vib_proj.
      - vib_proj: freeze backbone (+ periodic branch if any); train mu_layer, logvar_layer, proj
        (requires timer_vib=1).
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
    elif mode == "periodic_emb_proj":
        if not int(getattr(args, "periodic_embedding_branch", 0)):
            raise ValueError("finetune_trainable=periodic_emb_proj requires periodic_embedding_branch=1")
        for p in m.hour_embed.parameters():
            p.requires_grad = True
        for p in m.day_embed.parameters():
            p.requires_grad = True
        m.periodic_gamma.requires_grad = True
        align = getattr(m, "periodic_to_hidden", None)
        if align is not None:
            for p in align.parameters():
                p.requires_grad = True
        for p in m.backbone.proj.parameters():
            p.requires_grad = True
    elif mode == "proj_only":
        for p in m.backbone.proj.parameters():
            p.requires_grad = True
    elif mode == "vib_proj":
        if not int(getattr(args, "timer_vib", 0)):
            raise ValueError("finetune_trainable=vib_proj requires timer_vib=1")
        if getattr(m, "mu_layer", None) is not None:
            for p in m.mu_layer.parameters():
                p.requires_grad = True
        if getattr(m, "logvar_layer", None) is not None:
            for p in m.logvar_layer.parameters():
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