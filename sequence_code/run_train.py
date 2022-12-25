import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset, ClozeDataSet
from models import S3RecModel, BERT4Rec
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)

import wandb


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sweep", default="False", type=bool)
    
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers" # number of encoder blocks
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

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--patience", default=20, help="patience")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

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
    
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    
    """ SAS 종속
    """
    if "Finetune_full" == args.model_name : 
        args = {}
        args.item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

        args.user_seq, args.max_item, args.valid_rating_matrix, args.test_rating_matrix, _ 
        = get_user_seqs(
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
    
    elif( "BERT4REC" == args.model ):
        ############# 중요 #############
        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        df = pd.read_csv(args.data_file)

        item_ids = df['item'].unique()
        user_ids = df['user'].unique()
        num_item, num_user = len(item_ids), len(user_ids)
        num_batch = num_user // batch_size

        # user, item indexing
        item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
        user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

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
        
        seq_dataset = ClozeDataSet(user_train, num_user, num_item, max_len, mask_prob)
        train_dataset, eval_dataset = torch.tuils.data.dataset.random_split(dataset, [80,20])
        eval_dataset , test_dataset = torch.tuils.data.dataset.random_split(train_dataset, [10,10])

        train_data_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory = True)
        eval_data_loader = DataLoader(dataset=eval_dataset,batch_size=batch_size,shuffle=True,pin_memory = True)
        test_data_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,pin_memory = True)
    
    # train
    wandb.login()
    with wandb.init(project=f"Movie_Rec_{args.model_name}_train", config=vars(args)):
        if args.model_name == "ALL" :
            model = S3RecModel(args=args)

            trainer = FinetuneTrainer(
                model, train_dataloader, eval_dataloader, test_dataloader, None, args
            )
        elif args.model_name == "BERT4REC":
            model = BERT4RecModel(args = args) 
            
            trainer = Bert4RecTrainer(
                model, train_dataloader, eval_dataloader, test_dataloader, None, args
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
            
        if( "ALL" == args.model_name ):
            print("---------------Change to test_rating_matrix!-------------------")
            trainer.args.train_matrix = test_rating_matrix
            # load the best model
            # trainer.model.load_state_dict(torch.load(args.checkpoint_path))
            trainer.model.load_state_dict(torch.load(os.path.join(args.output_dir, checkpoint.split('/')[-1])))
            scores, result_info = trainer.test(0)
            wandb.log(result_info, step=epoch+1)
            print(result_info)
        else : 
            scores, result_info = trainer.test(0)
            print(scores )            
            


if __name__ == "__main__":
    main()
