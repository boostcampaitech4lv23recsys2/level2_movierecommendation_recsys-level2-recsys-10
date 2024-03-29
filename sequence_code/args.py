import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import PretrainDataset
from models import S3RecModel
from trainers import PretrainTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs_long,
    set_seed,
    modeling_sequence_bert,
    generate_item2idx,
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sweep", default="False", type=bool)
    parser.add_argument("--random_sort", default=0.0, type=float)
    parser.add_argument("--neg_from_pop", default=1, type=float)
    parser.add_argument("--loss_fn", default="cn", type=str)
    
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", type=str, default="BERT4Rec",
     choices=['Pretrain', 'Finetune_full', 'BERT4Rec'],
     help='학습 및 예측할 모델을 선택할 수 있습니다.'
     )

    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--max_len", default=350, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.0002570993420212356, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument(
        "--pre_epochs", type=int, default=10, help="number of pre_train epochs"
    )
    parser.add_argument("--pre_batch_size", type=int, default=512)

    parser.add_argument("--mask_p", type=float, default=0.25, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=1.5, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    
    parser.add_argument("--using_pretrain", action="store_true")

    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    # parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mask_prob", type=float, default=0.1456260638867132)
    
    args = parser.parse_args()

    return args
