import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset, ClozeDataSet
from models import S3RecModel, BERT4RecModel
from trainers import FinetuneTrainer, Bert4RecTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    indexinfo,
    AttributeDict,
    get_attr_seqs,
)

import wandb
import pandas as pd
from args import parse_args
from collections import defaultdict


def main(args):
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    
    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    class AttributeDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__


    elem =AttributeDict({})
    
    if "Finetune_full" == args.model_name : 

        args.item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
        elem.item2attribute, elem.attribute_size = get_item2attribute_json(args.item2attribute_file)

        item2idx_,idx2item_ = indexinfo.get_index_info()
        user_seq, elem.max_item, elem.train_matrix, elem.test_rating_matrix, elem.submissoin_rating_marix\
        = get_user_seqs(
            args.data_file,
            item2idx_
        )
        elem.valid_rating_matrix = elem.train_matrix
        elem.item_size = elem.max_item + 2
        elem.mask_id = elem.max_item + 1
        elem.attribute_size = elem.attribute_size + 1

        train_dataset = SASRecDataset(args,elem, user_seq, data_type="train")

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )

        eval_dataset = SASRecDataset(args,elem, user_seq, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
        )

        test_dataset = SASRecDataset(args,elem, user_seq, data_type="test")
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=args.batch_size
        )
    
    elif( "BERT4Rec" == args.model_name ):

        item2idx_,idx2item_ = indexinfo.get_index_info()
        user_seq, max_item, train_matrix,test_rating_matrix,submissoin_rating_marix = get_user_seqs(
            args.data_file,item2idx_, b_sort_by_time = True
        )

        attr_seq = get_attr_seqs(args.data_file,b_sort_by_time=True)
        
        elem.train_matrix = train_matrix
        elem.test_rating_matrix = test_rating_matrix
        elem.submission_rating_matrix = submissoin_rating_marix

        elem.num_user = len(user_seq)
        elem.num_item = max_item

        print(f'num users: {elem.num_user}, num items: {elem.num_item}')
        
        # max_len = 50
        # mask_prob = 0.2
        # hidden_units = 256
        # num_heads = 2
        # num_layers = 200
        # dropout_rate = 0.2
        # model setting
        args.max_len = 50 # 50
        args.hidden_units = 50 # 50
        args.num_heads = 1
        args.num_layers = 2
        args.dropout_rate=0.2
        args.num_workers = 1
        args.device = 'cuda' 

        # training setting
        args.lr = 0.002 # 0.002
        args.batch_size = 256
        args.num_epochs = 2000
        args.mask_prob = 0.15 # for cloze task
        args.criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시

        seq_dataset = ClozeDataSet(user_seq, attr_seq,args, elem)

        valid_size = int(len(seq_dataset) *0.2) # default val_ratio = 0.2
        train_size = len(seq_dataset) - valid_size
        train_dataset, eval_dataset = torch.utils.data.dataset.random_split(seq_dataset, [train_size,valid_size])

        eval_size = int(len(eval_dataset) *0.5) # default val_ratio = 0.2
        test_size = len(eval_dataset) - eval_size
        eval_dataset , test_dataset = torch.utils.data.dataset.random_split(eval_dataset, [eval_size,test_size])

        train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle=True, pin_memory = True)
        eval_dataloader = DataLoader(dataset  = eval_dataset , batch_size = args.batch_size, shuffle=True, pin_memory = True)
        test_dataloader = DataLoader(dataset  = test_dataset , batch_size = args.batch_size, shuffle=True, pin_memory = True)
    # train
    wandb.login()
    with wandb.init(project=f"Movie_Rec_{args.model_name}_train", config=vars(args)):
        if args.model_name == "Finetune_full" :
            model = S3RecModel(args=args,elem=elem)

            trainer = FinetuneTrainer(
                model, train_dataloader, eval_dataloader, test_dataloader, None, args, elem
            )
        elif args.model_name == "BERT4Rec":
            model = BERT4RecModel(args,elem)
            
            trainer = Bert4RecTrainer(
                model, train_dataloader, eval_dataloader, test_dataloader, None, args,elem
            )

        print(args.using_pretrain)
        if args.using_pretrain:
            pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
            try:
                trainer.load(pretrained_path)
                print(f"Load Checkpoint From {pretrained_path}!")

            except FileNotFoundError:
                print(f"{pretrained_path} Not Found! The Model is same as SASRec")
        else:
            print("Not using pretrained model. The Model is same as SASRec")

        # save model args
        args_str = f"{args.model_name}-{args.data_name}"
        args.log_file = os.path.join(args.output_dir, args_str + ".txt")
        print(str(args))

        # save model
        checkpoint = args_str + ".pt"
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            print('#####################################################################')
            trainer.train(epoch)

            scores, _ = trainer.valid(epoch)
            
            early_stopping(np.array(scores[-1:]), trainer.model)
            # early_stopping(np.array(scores[2]), trainer.model) # RECALL@10
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        if( "Finetune_full" == args.model_name ):
            print("---------------Change to test_rating_matrix!-------------------")
            trainer.args.train_matrix = elem.test_rating_matrix
            # load the best model
            # trainer.model.load_state_dict(torch.load(args.checkpoint_path))
            trainer.model.load_state_dict(torch.load(os.path.join(args.output_dir, checkpoint.split('/')[-1])))
        scores, result_info = trainer.test(0)
        wandb.log(result_info, step=epoch+1)
        print(result_info)


if __name__ == "__main__":
    args = parse_args()
    main(args)
