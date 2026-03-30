import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

from exp.exp_forecast import Exp_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from utils.tools import HiddenPrints, infer_resonance_period_hours_from_freq

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Large Time Series Model')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[forecast, imputation, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Timer',
                        help='model name, options: [Timer TrmEncoder]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)



    # model define
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='optimizer learning rate (Timer finetune: main group for non-ω/λ params, default 1e-4)',
    )
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt file')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
    parser.add_argument(
        '--finetune_trainable',
        type=str,
        default='full',
        choices=[
            'full',
            'last_layer',
            'resonance_only',
            'last_attention_only',
            'harmonic_proj',
        ],
        help='Timer: full; last_layer; resonance_only; last_attention_only; '
        'harmonic_proj=harmonic inner + proj only (needs harmonic_gated_resonance=1)',
    )
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--patch_len', type=int, default=24, help='input sequence length')
    parser.add_argument(
        '--diurnal_attn_bias',
        type=int,
        default=0,
        help='Timer only: 1 = add cos(2*pi*|i-j|/period) bias on selected heads; 0 = off (matches old ckpt)',
    )
    parser.add_argument(
        '--diurnal_lambda',
        type=float,
        default=1.0,
        help='Timer: strength of diurnal cos term (logits get scale*scores + lambda*cos after mask path)',
    )
    parser.add_argument(
        '--diurnal_period',
        type=float,
        default=24.0,
        help='Timer: cos period in patch-token units (e.g. 24 with patch_len=1 stride=1 on hourly data)',
    )
    parser.add_argument(
        '--resonance_last_layer',
        type=int,
        default=0,
        help='Timer: 1 = legacy last-layer multi-head resonance (ignored if harmonic_gated_resonance=1)',
    )
    parser.add_argument(
        '--harmonic_gated_resonance',
        type=int,
        default=0,
        help='Timer: 1 = harmonic gated specialist head (α·QK+(1-α)·bias)⊗σ(gate), K harmonics, O(L) trig',
    )
    parser.add_argument(
        '--harmonic_specialist_head',
        type=int,
        default=0,
        help='Head index (0..n_heads-1) for harmonic gated specialist; other heads vanilla',
    )
    parser.add_argument(
        '--harmonic_n_harmonics',
        type=int,
        default=3,
        help='Number of harmonics ω,2ω,.. in specialist bias/gate',
    )
    parser.add_argument(
        '--harmonic_lambda_init',
        type=float,
        default=1e-4,
        help='Initial |λ_k| per harmonic (strength warmup; multiplicative gate only)',
    )
    parser.add_argument(
        '--harmonic_fft_warmstart',
        type=int,
        default=1,
        help='Timer: 1 = first training batch sets ω from rFFT peak on raw series (harmonic only)',
    )
    parser.add_argument(
        '--resonance_head_mask',
        type=str,
        default='',
        help='Comma/space-separated 0/1 per head; empty = all heads use resonance',
    )
    parser.add_argument(
        '--resonance_dt_hours',
        type=float,
        default=1.0,
        help='Timer: hours per raw timestep for patch-center physical time T (e.g. 1 for hourly)',
    )
    parser.add_argument(
        '--resonance_lambda_init',
        type=float,
        default=0.1,
        help='Initial per-head resonance strength λ (learnable)',
    )
    parser.add_argument(
        '--resonance_phi_init',
        type=float,
        default=0.0,
        help='Initial per-head phase φ (learnable)',
    )
    parser.add_argument(
        '--resonance_omega_init',
        type=float,
        default=None,
        help='Override ω init; if unset, ω=1/resonance_period_hours (from --resonance_period_hours or --freq)',
    )
    parser.add_argument(
        '--resonance_period_hours',
        type=float,
        default=None,
        help='Physical cycle in hours for ω=1/period when resonance_omega_init unset; None→infer from --freq',
    )
    parser.add_argument(
        '--finetune_res_omega_lr',
        type=float,
        default=1e-5,
        help='Timer finetune: Adam lr for res_omega (scheduler keeps this fixed while main lr decays)',
    )
    parser.add_argument(
        '--finetune_res_lambda_lr',
        type=float,
        default=1e-5,
        help='Timer finetune: Adam lr for res_lambda',
    )
    parser.add_argument(
        '--loss_fft_alpha',
        type=float,
        default=0.0,
        help='Forecast finetune: Total=(1-α)*MSE+α*MAE(rFFT(pred),rFFT(target)); 0=pure MSE',
    )
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='mask ratio')
    parser.add_argument('--data_type', type=str, default='custom', help='data_type')

    parser.add_argument('--decay_fac', type=float, default=0.75)

    # cosin decay
    parser.add_argument('--cos_warm_up_steps', type=int, default=100)
    parser.add_argument('--cos_max_decay_steps', type=int, default=60000)
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10)
    parser.add_argument('--cos_max', type=float, default=1e-4)
    parser.add_argument('--cos_min', type=float, default=2e-6)

    # weight decay
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # autoregressive configs
    parser.add_argument('--use_ims', action='store_true', help='Iterated multi-step', default=False)
    parser.add_argument('--output_len', type=int, default=96, help='output len')
    parser.add_argument('--output_len_list', type=int, nargs="+", help="output_len_list")

    # train_test
    parser.add_argument('--train_test', type=int, default=1, help='train_test')
    parser.add_argument('--is_finetuning', type=int, default=1, help='status')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    args = parser.parse_args()
    if getattr(args, "resonance_period_hours", None) is None:
        args.resonance_period_hours = infer_resonance_period_hours_from_freq(args.freq)
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print(
            'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank,
                                                                                     gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)

    if args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'forecast':
        Exp = Exp_Forecast
    else:
        raise ValueError('task name not found')

    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        print(args)
        if args.is_finetuning:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    ii)
                setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.finetune(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
