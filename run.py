import os
import random
import torch
import numpy as np
import pandas as pd
import argparse

from engines.engine import Engine
from utils.logger import get_logger
from utils.utils import parsing_syntax, ConfigDict, load_config, update_config
import os

torch.set_num_threads(3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTime')

    
    parser.add_argument('--gpu', type=int, default=3, help='gpu id')
    parser.add_argument('--pretrain_model_path', type=str, default="checkpoints/checkpoint2036_['Taxi_NYC']_MM_lightning_lr0.00025_bs8_sl960_pl384_edim128_vs128_patch96_timestamp[1005_0719]/model_s2036.pth", help="pretrain for few shot")
    parser.add_argument('--eval_model_path', type=str, default="checkpoints/checkpoint2036_['gla']_MM_lightning_lr0.0004_bs2_sl96_pl24_edim32_vs32_patch24_timestamp[0517_1618]/model_s2036.pth", help='model evaluation')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--is_norm', type=int, default=0, help='norm')
    parser.add_argument('--seed', type=int, default=2036, help='random seed')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

    parser.add_argument('--config_filename', type=str, default='data_configs/roma.yaml', help='Configuration yaml file')
    parser.add_argument('--train_ratio', type=float, default=1.0, help="training data's ratio")

    # ablation switches
    parser.add_argument('--use_pmrl_svd', type=int, default=0)
    parser.add_argument('--lambda_svd', type=float, default=0.1, help='weight for PMRL SVD loss')
    parser.add_argument('--svd_tau1', type=float, default=0.07, help='temperature for singular value objective')
    parser.add_argument('--svd_tau2', type=float, default=0.07, help='temperature for principal direction regularization')

    parser.add_argument('--use_vot_dpe', type=int, default=0)
    parser.add_argument('--use_text_dpe', type=int, default=1, help='use text branch in DPE')
    parser.add_argument('--use_image_dpe', type=int, default=1, help='use image branch in DPE')
    parser.add_argument('--use_trend', type=int, default=1, help='use trend component in DPE')
    parser.add_argument('--use_seasonal', type=int, default=1, help='use seasonal component in DPE')
    parser.add_argument('--dpe_trend_queries', type=int, default=4, help='number of learnable queries for trend DPE')
    parser.add_argument('--dpe_seasonal_queries', type=int, default=8, help='number of learnable queries for seasonal DPE')
    parser.add_argument('--dpe_num_heads', type=int, default=4, help='number of DPE attention heads')

    parser.add_argument('--use_confu_loss', type=int, default=0)
    parser.add_argument('--lambda_confu', type=float, default=0.1)
    parser.add_argument('--confu_temperature', type=float, default=0.07)

    parser.add_argument('--disable_mm_selector', type=int, default=0)
    parser.add_argument('--force_all_modalities', type=int, default=0)

    
    args, unknown = parser.parse_known_args()
    unknown = parsing_syntax(unknown)

    config = load_config(args.config_filename)
    config = ConfigDict(config)
    config = update_config(config, unknown)
    for attr, value in config.items():
        setattr(args, attr, value)

    for key in [
        "use_pmrl_svd",
        "use_vot_dpe",
        "use_text_dpe",
        "use_image_dpe",
        "use_trend",
        "use_seasonal",
        "use_confu_loss",
        "disable_mm_selector",
        "force_all_modalities",
    ]:
        setattr(args, key, bool(getattr(args, key)))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    engine = Engine(args)
    if args.is_training == 1:
        engine.train()
    elif args.is_training == 0:
        engine.test()
    elif args.is_training == 2:
        engine.fine_tune()
    else:
        raise NotImplementedError