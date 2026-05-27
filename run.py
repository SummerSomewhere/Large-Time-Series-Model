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
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
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
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--patch_len', type=int, default=24, help='input sequence length')
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

    # Refinement mechanism
    parser.add_argument('--enable_refinement', type=int, default=0, help='enable refinement mechanism for last layer')
    parser.add_argument('--refine_patches', type=str, default='5,6',
                       help='comma-separated patch indices for refinement (0-indexed), e.g., "5,6" for patches 6,7')
    parser.add_argument('--refine_iterations', type=int, default=2, help='number of refinement iterations')
    parser.add_argument('--refine_alpha', type=float, default=0.5, help='refinement blending coefficient')
    parser.add_argument('--random_refine_patches', action='store_true', default=False,
                       help='randomly select patches for refinement instead of using MI-based selection')

    # Hierarchical Alignment Loss (Attn-MI Alignment)
    # Two modes:
    #   1. Post-softmax KL: L_align = (1/sum(M)) * sum_i M_i * KL(softmax(MI/tau) || softmax(attn_scores))
    #   2. Logit-space MSE (new): L_align = (1/sum(M)) * sum_i M_i * sum_h MSE(logit_{i}^{(h)}, gamma * P_MI^{(l)})
    parser.add_argument('--use_align_loss', action='store_true', default=False,
                       help='enable hierarchical alignment loss during fine-tuning')
    parser.add_argument('--align_loss_layers', type=str, default='0,1,2,3,4,5,6,7',
                       help='comma-separated layer indices to apply alignment loss (0-indexed)')
    parser.add_argument('--align_loss_weight', type=float, default=1.0,
                       help='weight beta for the alignment loss term')
    parser.add_argument('--align_loss_tau', type=float, default=0.1,
                       help='temperature for softmax on MI scores: P_MI = softmax(MI/tau)  [post-softmax mode only]')
    parser.add_argument('--align_loss_file', type=str, default='',
                       help='path to .json file containing pre-computed MI distributions per layer')
    parser.add_argument('--align_loss_freq', type=int, default=1,
                       help='compute alignment loss every N steps (1=every step, 4=every 4 steps)')
    parser.add_argument('--align_loss_mode', type=str, default='kl',
                       choices=['kl', 'logit_mse', 'softdtw'],
                       help="alignment loss formulation: 'kl' (post-softmax KL), "
                            "'logit_mse' (logit-space MSE), or 'softdtw' (elastic Soft-DTW)")
    parser.add_argument('--align_gamma', type=float, default=0.0,
                       help="gamma scaling coefficient for logit-space MSE mode. "
                            "0 means auto-set to sqrt(d_head). Can also be a learnable parameter.")
    parser.add_argument('--align_gamma_learnable', action='store_true', default=False,
                       help='make gamma a learnable parameter (logit_mse mode only)')
    parser.add_argument('--align_sharp_heads', type=int, default=0,
                       help='number of sharpest heads to average as the Logit target (0=all heads)')
    parser.add_argument('--align_dtw_gamma', type=float, default=1.0,
                       help="Soft-DTW temperature (gamma) for 'softdtw' mode. "
                            "Controls softness of the alignment. "
                            "gamma→0: hard DTW (sharp alignment). "
                            "gamma→inf: approaches MSE (no temporal warping). "
                            "Recommended: [0.01, 0.1, 1.0]. Default: 1.0")
    parser.add_argument('--align_dtw_bw', type=int, default=-1,
                       help="Sakoe-Chiba bandwidth for 'softdtw' mode. "
                            "-1 means full alignment matrix (no band constraint). "
                            "Positive int limits max deviation from diagonal. "
                            "Default: -1 (safe for short sequences, S~7)")
    parser.add_argument('--align_pool_size', type=int, default=1,
                       help="Pooling window size W for region-wise alignment. "
                            "Before computing the alignment loss, both the MI distribution "
                            "and attention distributions are pooled with AvgPool(W) to "
                            "coarsen the granularity from token-level to region-level. "
                            "W=1: no pooling (token-to-token, the default). "
                            "W=2-4: recommended for dense time series like ETTh1. "
                            "If S % W != 0, the last remainder tokens are dropped. "
                            "Default: 1 (disabled)")

    args = parser.parse_args()

    if hasattr(args, 'refine_patches') and isinstance(args.refine_patches, str):
        args.refine_patches = [int(x.strip()) for x in args.refine_patches.split(',') if x.strip()]
    if hasattr(args, 'align_loss_layers') and isinstance(args.align_loss_layers, str):
        args.align_loss_layers = [int(x.strip()) for x in args.align_loss_layers.split(',') if x.strip()]
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
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_al{}_all{}_alm{}_{}'.format(
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
                    int(args.use_align_loss),
                    '_'.join(map(str, args.align_loss_layers)),
                    args.align_loss_mode,
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
