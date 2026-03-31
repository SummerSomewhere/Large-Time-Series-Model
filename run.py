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
from utils.tools import HiddenPrints

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
        choices=['full', 'last_layer', 'periodic_emb_proj', 'proj_only', 'vib_proj'],
        help='Timer: full; last_layer; periodic_emb_proj=hour/day embed + gamma + proj only '
        '(needs periodic_embedding_branch=1); proj_only=patch Linear head only; '
        'vib_proj=VIB mu/logvar+proj only (needs timer_vib=1)',
    )
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--patch_len', type=int, default=24, help='input sequence length')
    parser.add_argument(
        '--periodic_embedding_branch',
        type=int,
        default=0,
        help='Timer: 1 = hour/day nn.Embedding residual + gamma before proj (uses batch_x_mark time features)',
    )
    parser.add_argument(
        '--periodic_emb_bank_dim',
        type=int,
        default=0,
        help='Timer: if >0, hour/day embeddings use this dim and a Linear maps to d_model; 0 = embed directly in d_model',
    )
    parser.add_argument(
        '--loss_fft_alpha',
        type=float,
        default=0.0,
        help='Forecast finetune: Total=(1-α)*MSE+α*MAE(rFFT(pred),rFFT(target)); 0=pure MSE',
    )
    parser.add_argument(
        '--timer_vib',
        type=int,
        default=0,
        help='Timer forecast: VIB (mu/logvar) before Linear proj; finetune loss += beta*KL(q(z|x)||N(0,I))',
    )
    parser.add_argument(
        '--vib_beta_max',
        type=float,
        default=1e-4,
        help='Timer VIB: target beta on KL after warmup (linear ramp over remaining steps)',
    )
    parser.add_argument(
        '--vib_warmup_ratio',
        type=float,
        default=0.1,
        help='Timer VIB: first this fraction of finetune steps use beta=0',
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

    # Timer: MI-guided representation recycle (forecast only) — isolated second TF on selected patches
    parser.add_argument(
        '--recycle_encoder_layers',
        type=str,
        default='',
        help='Timer: comma-separated 0-based layers for recycle (e.g. 0,7). Empty + single -1 uses last layer only.',
    )
    parser.add_argument(
        '--recycle_encoder_layer',
        type=int,
        default=-1,
        help='Timer: single layer index if recycle_encoder_layers empty; -1 disables',
    )
    parser.add_argument(
        '--recycle_patch_indices',
        type=str,
        default='',
        help='Timer: comma-separated input patch indices when no npy (e.g. 3 or 2,3)',
    )
    parser.add_argument(
        '--recycle_hsic_mean_npy',
        type=str,
        default='',
        help='Timer: optional hsic_mean.npy [L,T]; at layer l use row l for peak patches (else manual list)',
    )
    parser.add_argument(
        '--recycle_peak_mode',
        type=str,
        default='tukey',
        choices=['tukey', 'mean_iqr'],
        help='Timer: tukey=Q3+1.5*IQR; mean_iqr=mean+1.5*IQR on patch MI curve',
    )
    parser.add_argument(
        '--recycle_alpha',
        type=float,
        default=0.05,
        help='Timer: if recycle_round_alphas empty, one round uses this alpha',
    )
    parser.add_argument(
        '--recycle_round_alphas',
        type=str,
        default='',
        help='Timer: comma-separated alphas for successive rounds per patch (e.g. 0.4,0.2)',
    )

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    args = parser.parse_args()
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

    # Always visible on rank 0 (not swallowed by HiddenPrints for non-zero ranks).
    _lr_print = int(os.environ.get("LOCAL_RANK", "0"))
    if _lr_print == 0 and args.task_name == 'forecast':
        _ft = int(getattr(args, 'is_finetuning', 0) or 0)
        print(
            f"[run.py] is_finetuning={_ft} -> "
            f"{'exp.finetune() then exp.test() (best checkpoint.pth loaded at end of finetune)' if _ft else 'exp.test() only — weights from ckpt_path in model __init__, no finetune'}",
            flush=True,
        )

    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        print(args)
        if int(getattr(args, 'is_finetuning', 0) or 0):
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
