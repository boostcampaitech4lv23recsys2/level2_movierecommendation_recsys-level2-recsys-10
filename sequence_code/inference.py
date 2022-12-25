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
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
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
        ############# 중요 #############
        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        df = pd.read_csv(args.data_file)

        item_ids = df['item'].unique()
        user_ids = df['user'].unique()
        num_item, num_user = len(item_ids), len(user_ids)
        num_batch = num_user // args.batch_size

        # user, item indexing
        item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
        user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)
        args.item2idx = item2idx 

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
        del df['item'], df['user'] 

        # train set, valid set 생성
        users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
        user_train = {}
        user_valid = {}
        for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
            users[u].append(i)

        for user in users:
            user_train[user] = users[user]
            user_valid[user] = [users[user][-1]]

        print(f'num users: {num_user}, num items: {num_item}')  
        
        # max_len = 50
        # mask_prob = 0.2
        # hidden_units = 256
        # num_heads = 2
        # num_layers = 200
        # dropout_rate = 0.2
        # model setting
        max_len = 50 # 50
        hidden_units = 50 # 50
        num_heads = 1
        num_layers = 2
        dropout_rate=0.5
        num_workers = 1
        device = 'cuda' 

        # training setting
        lr = 0.001
        batch_size = 128
        num_epochs = 200
        mask_prob = 0.15 # for cloze task
        
        submission_dataset = ClozeDataSet(user_train, num_user, num_item, max_len, mask_prob, is_submission = True)
        submission_sampler = SequentialSampler(submission_dataset)
        submission_dataloader = DataLoader(
            submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
        )
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시

        model = BERT4RecModel(num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate)
        trainer = Bert4RecTrainer(
                model,  None, None, None, submission_dataloader, args, criterion
        )
    trainer.load(args.checkpoint_path)
    print(f"Load model from {args.checkpoint_path} for submission!")
    
    preds = trainer.submission(0)
    generate_submission_file(args.data_file, preds)


if __name__ == "__main__":
    args = parse_args()
    main(args)
