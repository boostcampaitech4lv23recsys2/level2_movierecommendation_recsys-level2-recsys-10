import argparse
import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from  torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader#, RandomSampler, SequentialSampler
import wandb


from preprocessing import numerize
from datasets import (
    MultiVAEDataset, MultiVAEValidDataset,
    RecVAEDataset, RecVAEValidDataset,
    )
# from data_loader import DataLoader
from trainer import Trainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    NDCG_binary_at_k_batch,
    Recall_at_k_batch
)
from models import (
    MultiVAE,
    MultiDAE,
    # EASE,
    RecVAE
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--train_data", default="train_ratings.csv", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="multiVAE", type=str)
    ## sequential model args
    # parser.add_argument(
    #     "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    # )
    # parser.add_argument(
    #     "--num_hidden_layers", type=int, default=2, help="number of layers"
    # )
    # parser.add_argument("--num_attention_heads", default=2, type=int)
    # parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    # parser.add_argument(
    #     "--attention_probs_dropout_prob",
    #     type=float,
    #     default=0.5,
    #     help="attention dropout p",
    # )
    # parser.add_argument(
    #     "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    # )
    # parser.add_argument("--initializer_range", type=float, default=0.02)
    # parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--lr_decay_step", type=int, default=1000, help="default: 1000") 
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--batch_size", type=int, default=250, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument("--log_freq", type=int, default=100, metavar='N', help="per epoch print res")
    parser.add_argument("--reg", type=int, default=750, help="reg for EASE")
    parser.add_argument("--genre_filter", type=bool, default=False, help=" ")
    parser.add_argument('--hidden_dim', type=int, default=600)
    parser.add_argument('--latent_dim', type=int, default=200)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.005)

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    
    # parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    # parser.add_argument("--using_pretrain", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    args.device = torch.device("cuda" if args.cuda else "cpu")

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # args.data_file = args.data_dir + "train_ratings.csv"
    # item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    # user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
    #     args.data_file
    # )

    # item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    # args.item_size = max_item + 2
    # args.mask_id = max_item + 1
    # args.attribute_size = attribute_size + 1

    print("Load and Preprocess dataset")
    print("model: ", args.model_name)
    if args.model_name == 'multiVAE':
        train_dataset = MultiVAEDataset(args)
        valid_dataset = MultiVAEValidDataset(train_dataset = train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False)

        # 모델 정의
        p_dims=[200, 600, train_dataset.n_items]
        model = MultiVAE(p_dims).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    elif args.model_name == 'multiDAE':
        train_dataset = MultiVAEDataset(args)
        valid_dataset = MultiVAEValidDataset(train_dataset = train_dataset)
       
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False)

        # 모델 정의
        p_dims=[200, 600, train_dataset.n_items]
        model = MultiDAE(p_dims).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    elif args.model_name == "EASE":
        pass
    elif args.model_name == "recVAE":
        train_dataset = RecVAEDataset(args)
        valid_dataset = RecVAEValidDataset(train_dataset = train_dataset, genre_filter = args.genre_filter)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False)
        
        # 모델 정의
        model = RecVAE(
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            input_dim=train_dataset.n_items,
            dropout_rate=args.dropout_rate
        ).to(args.device)

        optimizer_encoder = torch.optim.Adam(set(model.decoder.parameters()), lr=args.lr)
        optimizer_decoder = torch.optim.Adam(set(model.encoder.parameters()), lr=args.lr)
        optimizer = [optimizer_encoder, optimizer_decoder]

        # -- Learning Rate Scheduler
        scheduler_encoder = StepLR(optimizer_encoder, step_size=args.lr_decay_step, gamma=0.5)
        scheduler_decoder = StepLR(optimizer_decoder, step_size=args.lr_decay_step, gamma=0.5)

    
    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    checkpoint = args_str + ".pt"
    args.save = os.path.join(args.output_dir, checkpoint)

    trainer = Trainer(
        model, optimizer, train_loader, valid_loader, args
    )

    ###############################################################################
    # Training code
    ###############################################################################

    wandb.login()
    with wandb.init(project="MovieRec_encoder", config=vars(args)):
        best_r10 = -np.inf

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time()
            trainer.train(epoch, args.model_name)
            total_loss, r10, r20 = trainer.evaluate(args.model_name) # mode: valid
            # val_loss, n100, r20, r50 = evaluate(model, criterion, vad_data_tr, vad_data_te, is_VAE=False)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                        'r10 {:5.3f} | r20 {:5.3f}'.format(
                            epoch, time() - epoch_start_time, total_loss, r10, r20))
            print('-' * 89)

            # Save the model if the n100 is the best we've seen so far.
            if r10 > best_r10:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
            best_r10 = r10
            wandb.log({
                "epoch": epoch,
                "train_loss": total_loss,
                "recall@10": r10,
                "recall@20": r20
            })


if __name__ == "__main__":
    main()
