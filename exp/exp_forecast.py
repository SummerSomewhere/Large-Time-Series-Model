from __future__ import annotations

import json
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import (
    BetaScheduler,
    EarlyStopping,
    apply_timer_finetune_freeze,
    attn_map,
    build_timer_finetune_param_groups,
    fft_complex_mae,
    LargeScheduler,
    visual,
)

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Timer: sig_gate.alpha (and HM-ISR / CAB scalars) are split into high-lr group via
        # build_timer_finetune_param_groups — same as filter(requires_grad) but explicit groups.
        if getattr(self.args, "model", None) == "Timer":
            groups = build_timer_finetune_param_groups(self.model, self.args)
            if groups:
                return optim.Adam(groups)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "No trainable parameters (check finetune_trainable / freeze settings)."
            )
        if self.args.use_weight_decay:
            model_optim = optim.Adam(
                params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
            )
        else:
            model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    _sig_gate_debug_bwd_prints_left: int | None = None

    def _debug_sig_gate_after_backward(self) -> None:
        """After loss.backward(): print sig_gate.alpha.grad (first N steps if SIG_REFINER_DEBUG=1)."""
        if int(os.environ.get("SIG_REFINER_DEBUG", "0")) == 0:
            return
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        if getattr(self.args, "model", None) != "Timer":
            return
        rem = Exp_Forecast._sig_gate_debug_bwd_prints_left
        if rem is None:
            Exp_Forecast._sig_gate_debug_bwd_prints_left = 10
            rem = 10
        if rem <= 0:
            return
        Exp_Forecast._sig_gate_debug_bwd_prints_left = rem - 1
        m = self.model.module if hasattr(self.model, "module") else self.model
        sg = getattr(m, "sig_gate", None)
        if sg is None:
            print(">>> DEBUG [after bwd]: sig_gate is None (use --sig_gate 1; HM-ISR replaces it)", flush=True)
            return
        a = sg.alpha
        if a.grad is None:
            print(
                ">>> DEBUG [after bwd]: sig_gate.alpha.grad is None (check graph / find_unused_parameters)",
                flush=True,
            )
        else:
            print(
                f">>> DEBUG [after bwd]: sig_gate.alpha grad mean={a.grad.mean().item():.8e}",
                flush=True,
            )

    def _timer_mi_preservation_active(self) -> bool:
        return getattr(self.args, "model", None) == "Timer" and int(
            getattr(self.args, "mi_preservation", 0)
        ) == 1

    def _finetune_forecast_loss(self, criterion, outputs, batch_y, flag: str):
        """TimeEmb-style: (1-α)*MSE + α*mean(|rFFT(pred)-rFFT(target)|) (complex coeff MAE)."""
        alpha = float(getattr(self.args, "loss_fft_alpha", 0.0))
        if self.args.use_ims:
            pred = outputs[:, -self.args.seq_len:, :]
            true = batch_y
            if flag == "test":
                pred = pred[:, -self.args.pred_len:, :]
                true = true[:, -self.args.pred_len:, :]
        else:
            pred = outputs[:, -self.args.pred_len:, :]
            true = batch_y[:, -self.args.pred_len:, :]
        mse = criterion(pred, true)
        if alpha <= 0.0:
            return mse
        fft_term = fft_complex_mae(pred, true, time_dim=1)
        if alpha >= 1.0:
            return fft_term
        return (1.0 - alpha) * mse + alpha * fft_term

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _print_sig_gate_alpha_stats(self) -> None:
        """Print HM-ISR and/or legacy SIG-Gate alpha stats (rank 0 only)."""
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        m = self.model.module if hasattr(self.model, "module") else self.model

        def _print_hm(tag: str, mod) -> None:
            a = mod.alpha.detach().float().cpu().view(-1).numpy()
            ls = float(mod.lambda_scale.detach().cpu().item())
            mask = mod.patch_hard_mask.detach().float().cpu().view(-1).numpy()
            print(f"{tag}.lambda_scale={ls}", flush=True)
            print(
                f"{tag}.hard_mask: "
                + json.dumps({int(i): float(mask[i]) for i in range(int(mask.size))}, ensure_ascii=False),
                flush=True,
            )
            if a.size > 6:
                print(
                    f"{tag}.alpha[3]={float(a[3]):.6f} {tag}.alpha[6]={float(a[6]):.6f}",
                    flush=True,
                )
            print(
                f"{tag}.alpha: n_patches={a.size} min={float(a.min()):.6f} max={float(a.max()):.6f} "
                f"mean={float(a.mean()):.6f} std={float(a.std()):.6f}",
                flush=True,
            )
            flat = a.flatten().tolist()
            print(
                f"{tag}.alpha per-patch: "
                + json.dumps({i: float(flat[i]) for i in range(len(flat))}, ensure_ascii=False),
                flush=True,
            )

        he = getattr(m, "hm_isr_entry", None)
        hx = getattr(m, "hm_isr_exit", None)
        if he is not None:
            _print_hm("hm_isr_entry", he)
        if hx is not None:
            _print_hm("hm_isr_exit", hx)

        sg = getattr(m, "sig_gate", None)
        if sg is None:
            return
        a = sg.alpha.detach().float().cpu().view(-1).numpy()
        _ls = float(sg.lambda_scale.detach().cpu().item()) if hasattr(sg, "lambda_scale") else float("nan")
        print(f"sig_gate.lambda_scale={_ls}", flush=True)
        print(
            f"sig_gate.alpha: n_patches={a.size} min={float(a.min()):.6f} max={float(a.max()):.6f} "
            f"mean={float(a.mean()):.6f} std={float(a.std()):.6f}",
            flush=True,
        )
        flat = a.flatten().tolist()
        print(f"sig_gate.alpha per-patch (index: value): { {i: float(flat[i]) for i in range(len(flat))} }", flush=True)

    def _log_sig_gate_alpha_epoch(self, epoch_idx: int) -> None:
        """After each finetune epoch, print raw alpha vectors for HM-ISR / SIG-Gate (rank 0 only)."""
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        m = self.model.module if hasattr(self.model, "module") else self.model
        ep = epoch_idx + 1
        he = getattr(m, "hm_isr_entry", None)
        hx = getattr(m, "hm_isr_exit", None)
        sg = getattr(m, "sig_gate", None)
        if he is None and hx is None and sg is None:
            return
        if he is not None:
            print(f"epoch {ep} hm_isr_entry.alpha (flat): {he.alpha.data.detach().float().cpu().flatten().numpy()}", flush=True)
        if hx is not None:
            print(f"epoch {ep} hm_isr_exit.alpha (flat): {hx.alpha.data.detach().float().cpu().flatten().numpy()}", flush=True)
        if sg is not None:
            print(f"epoch {ep} sig_gate.alpha (flat): {sg.alpha.data.detach().float().cpu().flatten().numpy()}", flush=True)

    def _save_sig_gate_alpha_bar_chart(self, setting: str, save_dir: str) -> None:
        """
        After test metrics: bar charts of per-patch alpha for HM-ISR entry/exit and/or legacy sig_gate.
        """
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        m = self.model.module if hasattr(self.model, "module") else self.model
        data_name = getattr(self.args, "data", "ETTh1")
        specs = [
            ("hm_isr_entry", getattr(m, "hm_isr_entry", None), f"HM-ISR entry alpha ({data_name})", "hm_isr_entry_alpha_bar.png"),
            ("hm_isr_exit", getattr(m, "hm_isr_exit", None), f"HM-ISR exit alpha ({data_name})", "hm_isr_exit_alpha_bar.png"),
            ("sig_gate", getattr(m, "sig_gate", None), f"SIG-Gate Patch-wise Importance ({data_name})", "sig_gate_alpha_bar.png"),
        ]
        any_plot = False
        for _tag, mod, title, fig_name in specs:
            if mod is None:
                continue
            any_plot = True
            a = mod.alpha.detach().float().cpu().view(-1).numpy()
            n = int(a.size)
            indices = list(range(n))
            patch_dict = {int(i): float(a[i]) for i in range(n)}
            print(
                f"{_tag} alpha (post-test, per patch):\n"
                + json.dumps(patch_dict, indent=2, ensure_ascii=False),
                flush=True,
            )
            plt.figure(figsize=(max(6, n * 0.6), 4))
            plt.bar(indices, a.tolist(), color="steelblue", edgecolor="navy", linewidth=0.5)
            plt.xlabel("Patch Index")
            plt.ylabel("Alpha")
            plt.title(title)
            plt.xticks(indices)
            plt.grid(axis="y", linestyle="--", alpha=0.35)
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            out1 = os.path.join(save_dir, fig_name)
            plt.savefig(out1, dpi=150)
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            if os.path.isdir(ckpt_dir):
                out2 = os.path.join(ckpt_dir, fig_name)
                plt.savefig(out2, dpi=150)
                print(f"{_tag} alpha bar chart also saved to {out2}", flush=True)
            plt.close()
            print(f"{_tag} alpha bar chart saved to {out1}", flush=True)
        if not any_plot:
            return

    def _timer_vib_active(self) -> bool:
        return getattr(self.args, "model", None) == "Timer" and int(getattr(self.args, "timer_vib", 0))

    def _forecast_model_kwargs(self, return_vib_kl: bool, mi_on: bool) -> dict:
        """Timer-only keys; other backbones must not see unknown forward kwargs."""
        kw: dict = {}
        if return_vib_kl:
            kw["return_vib_kl"] = True
        if getattr(self.args, "model", None) == "Timer" and mi_on:
            kw["return_mi_feats"] = True
        return kw

    def _forward_forecast(
        self,
        batch_x,
        batch_x_mark,
        dec_inp,
        batch_y_mark,
        return_vib_kl: bool = False,
        return_mi_feats: bool = False,
    ):
        """
        Returns (outputs, vib_kl, attns, feat_early, feat_late, mi_loss).
        mi_loss is a scalar tensor from Timer.forward when mi_preservation+return_mi_feats (else None).
        """
        mi_on = bool(return_mi_feats) and self._timer_mi_preservation_active()
        if getattr(self.args, "model", None) != "Timer":
            mi_on = False
        fkw = self._forecast_model_kwargs(return_vib_kl, mi_on)

        if not self._timer_vib_active() or not return_vib_kl:
            if self.args.output_attention:
                out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **fkw)
                if mi_on:
                    dec_out, attns, fe, fl, mi_loss = out
                    return dec_out, None, attns, fe, fl, mi_loss
                dec_out, attns = out
                return dec_out, None, attns, None, None, None
            out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **fkw)
            if mi_on:
                dec_out, fe, fl, mi_loss = out
                return dec_out, None, None, fe, fl, mi_loss
            return out, None, None, None, None, None
        if self.args.output_attention:
            out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **fkw)
            if mi_on:
                dec_out, attns, vib_kl, fe, fl, mi_loss = out
                return dec_out, vib_kl, attns, fe, fl, mi_loss
            dec_out, attns, vib_kl = out
            return dec_out, vib_kl, attns, None, None, None
        out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **fkw)
        if mi_on:
            dec_out, vib_kl, fe, fl, mi_loss = out
            return dec_out, vib_kl, None, fe, fl, mi_loss
        dec_out, vib_kl = out
        return dec_out, vib_kl, None, None, None, None

    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali', vib_beta: float | None = None):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                outputs, vib_kl, _, _, _, _ = self._forward_forecast(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    return_vib_kl=self._timer_vib_active(),
                )

                loss = self._finetune_forecast_loss(criterion, outputs, batch_y, flag)
                if vib_kl is not None and vib_beta is not None:
                    loss = loss + float(vib_beta) * vib_kl

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def finetune(self, setting):
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        if train_steps <= 0:
            raise RuntimeError(
                "finetune: train DataLoader has zero batches (cannot finetune). "
                "Check batch_size / dataset / DistributedSampler."
            )
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, local_rank=self.args.local_rank
        )

        apply_timer_finetune_freeze(self.model, self.args)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        _n_all = sum(p.numel() for p in self.model.parameters())
        _n_tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model parameters (total): ", _n_all)
        print("Model parameters (trainable): ", _n_tr)
        if _n_tr <= 0:
            raise RuntimeError("finetune: no trainable parameters after apply_timer_finetune_freeze.")
        if int(self.args.finetune_epochs) <= 0:
            raise RuntimeError("finetune: finetune_epochs must be >= 1 (otherwise no training / no checkpoint).")
        scheduler = LargeScheduler(self.args, model_optim)

        total_ft_steps = max(1, self.args.finetune_epochs * train_steps)
        beta_sched = BetaScheduler(
            total_ft_steps,
            warmup_ratio=float(getattr(self.args, "vib_warmup_ratio", 0.1)),
            beta_max=float(getattr(self.args, "vib_beta_max", 1e-4)),
        )
        global_step = 0

        for epoch in range(self.args.finetune_epochs):
            iter_count = 0

            _dev = next(self.model.parameters()).device
            loss_val = torch.tensor(0., device=_dev)
            count = torch.tensor(0., device=_dev)

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs, vib_kl, _, _, _, mi_loss = self._forward_forecast(
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    return_vib_kl=self._timer_vib_active(),
                    return_mi_feats=self._timer_mi_preservation_active(),
                )
                beta_t = beta_sched(global_step) if self._timer_vib_active() else 0.0

                loss = self._finetune_forecast_loss(
                    criterion, outputs, batch_y, "vali"
                )
                if vib_kl is not None:
                    loss = loss + beta_t * vib_kl
                # Always add w*mi_loss when MI is on so mi_proj stays in the autograd graph under DDP
                # (warmup uses w=0 so no gradient to MI head until mi_warmup_epochs).
                if self._timer_mi_preservation_active() and mi_loss is not None:
                    w = float(getattr(self.args, "lambda_mi", 0.001))
                    if epoch < int(getattr(self.args, "mi_warmup_epochs", 0)):
                        w = 0.0
                    loss = loss + w * mi_loss

                loss_val += loss.item()
                count += 1
                global_step += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    _mem_a = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                    _mem_r = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
                    _mem_c = (
                        torch.cuda.memory_cached() / 1024 / 1024
                        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_cached")
                        else 0
                    )
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | beta: {3:.2e} | cost_time: {4:.0f} | memory: allocated {5:.0f}MB, reserved {6:.0f}MB, cached {7:.0f}MB "
                        .format(i, epoch + 1, loss.item(), beta_t, cost_time, _mem_a, _mem_r, _mem_c))
                    time_now = time.time()

                loss.backward()
                self._debug_sig_gate_after_backward()
                model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vib_beta_val = beta_sched(max(0, global_step - 1)) if self._timer_vib_active() else None
            vali_loss = self.vali(vali_data, vali_loader, criterion, vib_beta=vib_beta_val)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test', vib_beta=vib_beta_val)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            self._log_sig_gate_alpha_epoch(epoch)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(
                f"finetune: expected best checkpoint at {best_model_path} but file is missing. "
                "Training loop may not have saved (bug) or finetune_epochs was 0."
            )
        state = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self._finetune_checkpoint_path = os.path.abspath(best_model_path)
        print(
            f"Finetune: loaded best state_dict into self.model for testing "
            f"(trained_batches={global_step}, path={self._finetune_checkpoint_path})",
            flush=True,
        )

        return self.model

    def test(self, setting, test=0):

        _ck = getattr(self, "_finetune_checkpoint_path", None)
        if _ck:
            print(
                f"test(): using in-memory weights after finetune (best ckpt: {_ck})",
                flush=True,
            )
        else:
            print(
                "test(): using in-memory weights from model __init__ (no finetune in this process)",
                flush=True,
            )
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        self._print_sig_gate_alpha_stats()
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        preds_list = [[] for _ in range(len(self.args.output_len_list))]
        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()

        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    inference_steps = self.args.output_len // self.args.pred_len
                    dis = self.args.output_len - inference_steps * self.args.pred_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            tmp = batch_y_mark[:, j - 1:j, :]
                            batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                        if self.args.output_attention:
                            outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred_y.append(outputs[:, -self.args.pred_len:, :])
                    pred_y = torch.cat(pred_y, dim=1)

                    if dis != 0:
                        pred_y = pred_y[:, :-self.args.pred_len+dis, :]

                    if self.args.use_ims:
                        batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                            self.device)
                    else:
                        batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
                    outputs = pred_y.detach().cpu()
                    batch_y = batch_y.detach().cpu()

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds_list[output_ptr].append(pred)
                    trues_list[output_ptr].append(true)
                    if i % 10 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)

                        if self.args.local_rank == 0:
                            if self.args.output_attention:
                                attn = attns[0].cpu().numpy()[0, 0, :, :]
                                attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))

                            visual(gt, pd, os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))

        if self.args.output_len_list is not None:
            metrics_records = []
            for i in range(len(preds_list)):
                preds = preds_list[i]
                trues = trues_list[i]
                preds = torch.cat(preds, dim=0).numpy()
                trues = torch.cat(trues, dim=0).numpy()
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                print(f"output_len: {self.args.output_len_list[i]}")

                print('mse:{}, mae:{}'.format(mse, mae))
                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('\n')
                f.write('\n')
                f.close()
                metrics_records.append(
                    {
                        "output_len": int(self.args.output_len_list[i]),
                        "mse": float(mse),
                        "mae": float(mae),
                    }
                )

            metrics_path = os.environ.get("FORECAST_TEST_METRICS_JSON", "").strip()
            if metrics_path and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                _mdir = os.path.dirname(os.path.abspath(metrics_path))
                if _mdir:
                    os.makedirs(_mdir, exist_ok=True)
                try:
                    with open(metrics_path, "w", encoding="utf-8") as mf:
                        json.dump(metrics_records, mf, indent=2)
                except OSError as e:
                    print(f"WARNING: could not write FORECAST_TEST_METRICS_JSON={metrics_path}: {e}")

            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                self._save_sig_gate_alpha_bar_chart(setting, folder_path)

        return
