import argparse
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler

from datasets import SASRecDataset,ClozeDataSet
from models import S3RecModel,BERT4RecModel
from trainers import FinetuneTrainer,Bert4RecTrainer
from utils import (
    check_path,
    generate_submission_file,
    generate_submission_file_v2,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    indexinfo,
    AttributeDict,
)

from args import parse_args
import pandas as pd
import numpy as np
from collections import defaultdict

def main(args):
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"

    args_str = f"{args.model_name}-{args.data_name}"
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    
    elem =AttributeDict({})
    """SASRec 종속 
    """
    if "Finetune_full"== args.model_name : 
        item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

        user_seq, max_item, _, _, submission_rating_matrix = get_user_seqs(args.data_file)

        item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

        args.item_size = max_item + 2
        args.mask_id = max_item + 1
        args.attribute_size = attribute_size + 1

        # save model args
        print(str(args))

        args.item2attribute = item2attribute

        args.train_matrix = submission_rating_matrix

        submission_dataset = SASRecDataset(args, user_seq, data_type="submission")
        submission_sampler = SequentialSampler(submission_dataset)
        submission_dataloader = DataLoader(
            submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
        )

        model = S3RecModel(args=args)

        trainer = FinetuneTrainer(model, None, None, None, submission_dataloader, args)
    else : 
        item2idx_,idx2item_ = indexinfo.get_index_info()
        user_seq, max_item, train_matrix,test_rating_matrix,submissoin_rating_marix = get_user_seqs(
            args.data_file,item2idx_, b_sort_by_time = True
        )
        
        elem.train_matrix = train_matrix
        elem.test_rating_matrix = test_rating_matrix
        elem.submission_rating_matrix = submissoin_rating_marix

        elem.num_user = len(user_seq)
        elem.num_item = max_item

        print(f'num users: {elem.num_user}, num items: {elem.num_item}')
        
        # model setting
        args.max_len = 50 # 200
        args.hidden_units = 50 # 200
        args.num_heads = 1
        args.num_layers = 2
        args.dropout_rate = 0.5 # 0.2
        args.num_workers = 1
        args.device = 'cuda' 

        # training setting
        args.lr = 0.001
        args.batch_size = 128 # 256
        args.num_epochs = 200
        args.mask_prob = 0.15 # for cloze task
        
        submission_dataset = ClozeDataSet(user_seq, args, elem, is_submission=True)
        submission_sampler = SequentialSampler(submission_dataset)
        submission_dataloader = DataLoader(
            submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
        )
        args.criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
        model = BERT4RecModel(args,elem)
        trainer = Bert4RecTrainer(
                model,  None, None, None, submission_dataloader, args,elem
        )
    trainer.load(args.checkpoint_path)
    print(f"Load model from {args.checkpoint_path} for submission!")
    
    preds = trainer.submission(0)
    generate_submission_file_v2(args.data_file, preds,idx2item_)


if __name__ == "__main__":
    args = parse_args()
    main(args)
