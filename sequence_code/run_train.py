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
    
    if "Finetune_full" == args.model_name : 
        args = {}
        args.item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

        args.user_seq, args.max_item, args.valid_rating_matrix, args.test_rating_matrix, _ = get_user_seqs(
            args.data_file
        )

        args.item2attribute, args.attribute_size = get_item2attribute_json(item2attribute_file)

        args.item_size = max_item + 2
        args.mask_id = max_item + 1
        args.attribute_size = attribute_size + 1

        # save model args
        args_str = f"{args.model_name}-{args.data_name}"
        args.log_file = os.path.join(args.output_dir, args_str + ".txt")
        print(str(args))

        args.item2attribute = item2attribute
        # set item score in train set to `0` in validation
        args.train_matrix = valid_rating_matrix

        # save model
        checkpoint = args_str + ".pt"
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        train_dataset = SASRecDataset(args, user_seq, data_type="train")
        train_sampRandomSamplerler = (train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )

        eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
        )

        test_dataset = SASRecDataset(args, user_seq, data_type="test")
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=args.batch_size
        )
    
    elif( "BERT4Rec" == args.model_name ):
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
        df.sort_values(['user_idx', 'time'], inplace=True)
        del df['item'], df['user'] 

        # train set, valid set 생성
        users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
        user_train = {}
        user_valid = {}
        for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
            users[u].append(i)

        for user in users:
            user_train[user] = users[user][:-1]
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
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시

        seq_dataset = ClozeDataSet(user_train, num_user, num_item, max_len, mask_prob)
        valid_size = int(len(seq_dataset) *0.2) # default val_ratio = 0.2
        train_size = len(seq_dataset) - valid_size
        train_dataset, eval_dataset = torch.utils.data.dataset.random_split(seq_dataset, [train_size,valid_size])

        eval_size = int(len(eval_dataset) *0.5) # default val_ratio = 0.2
        test_size = len(eval_dataset) - eval_size
        eval_dataset , test_dataset = torch.utils.data.dataset.random_split(eval_dataset, [eval_size,test_size])

        train_dataloader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,pin_memory = True)
        eval_dataloader = DataLoader(dataset=eval_dataset,batch_size  =args.batch_size,shuffle=True,pin_memory = True)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size  =args.batch_size,shuffle=True,pin_memory = True)
    # train
    # wandb.login()
    # with wandb.init(project=f"Movie_Rec_{args.model_name}_train", config=vars(args)):
    if args.model_name == "ALL" :
        model = S3RecModel(args=args)

        trainer = FinetuneTrainer(
            model, train_dataloader, eval_dataloader, test_dataloader, None, args
        )
    elif args.model_name == "BERT4Rec":
        model = BERT4RecModel(num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate)
        
        trainer = Bert4RecTrainer(
            model, train_dataloader, eval_dataloader, test_dataloader, None, args, criterion
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
        trainer.args.train_matrix = test_rating_matrix
        # load the best model
        # trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        trainer.model.load_state_dict(torch.load(os.path.join(args.output_dir, checkpoint.split('/')[-1])))
    scores, result_info = trainer.test(0)
    # wandb.log(result_info, step=epoch+1)
    print(result_info)


if __name__ == "__main__":
    args = parse_args()
    main(args)
