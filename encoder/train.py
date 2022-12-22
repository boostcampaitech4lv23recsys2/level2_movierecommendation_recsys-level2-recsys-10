import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from preprocessing import preprocessing
from datasets import AEDatasets, multiVAEDatasets, MatrixDataset
from data_loader import DataLoader
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
    EASE,
    recVAE
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
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
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument("--log_freq", type=int, default=100, metavar='N', help="per epoch print res")
    parser.add_argument("--reg", type=int, default=750, help="reg for EASE")
     
    
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
    ###############################################################################
    # Load data
    ###############################################################################

    loader = DataLoader(args.data_dir)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')

    N = train_data.shape[0]
    # idxlist = list(range(N))

    ###############################################################################
    # Build the model
    ###############################################################################

    p_dims = [200, 600, n_items]

    # print(args.model_name)
    # model = MultiDAE(p_dims).to(device)
    # criterion = loss_function_vae
    # is_VAE=True
    if args.model_name == 'multiVAE':
        model = MultiVAE(p_dims).to(args.device)
        criterion = loss_function_vae
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        is_VAE=True
    elif args.model_name == 'multiDAE':
        model = MultiDAE(p_dims).to(args.device)
        criterion = loss_function_dae
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        is_VAE=False
    elif args.model_name == "EASE":
        # make_matrix_data_set = MatrixDataset()
        # user_train, user_valid = make_matrix_data_set.get_train_valid_data()
        # # X = make_matrix_data_set.make_sparse_matrix()
        
        # X_test = make_matrix_data_set.make_sparse_matrix(test = True)
        # model = EASE(X = X_test, reg = args.reg)
        # model.fit()
        pass
    elif args.model_name == "recVAE":
        pass
    

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    checkpoint = args_str + ".pt"
    args.save = os.path.join(args.output_dir, checkpoint)

    trainer = Trainer(
        model, criterion, optimizer, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, None, args
    )

    ###############################################################################
    # Training code
    ###############################################################################

    best_n100 = -np.inf
    update_count = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        trainer.train(epoch, is_VAE)
        val_loss, n100, r20, r50 = trainer.evaluate(is_VAE) # mode: valid
        # val_loss, n100, r20, r50 = evaluate(model, criterion, vad_data_tr, vad_data_te, is_VAE=False)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    n100, r20, r50))
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))


        # Save the model if the n100 is the best we've seen so far.
        if n100 > best_n100:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_n100 = n100

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, n100, r20, r50 = trainer.evaluate(is_VAE) # mode: test
    print('=' * 89)
    print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | '
            'r50 {:4.2f}'.format(test_loss, n100, r20, r50))
    print('=' * 89)


    # train_dataset = SASRecDataset(args, user_seq, data_type="train")
    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(
    #     train_dataset, sampler=train_sampler, batch_size=args.batch_size
    # )

    # eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    # )

    # test_dataset = SASRecDataset(args, user_seq, data_type="test")
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(
    #     test_dataset, sampler=test_sampler, batch_size=args.batch_size
    # )

    # args.item2attribute = item2attribute
    # # set item score in train set to `0` in validation
    # args.train_matrix = valid_rating_matrix



    # train_dataset = SASRecDataset(args, user_seq, data_type="train")
    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(
    #     train_dataset, sampler=train_sampler, batch_size=args.batch_size
    # )

    # eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    # )

    # test_dataset = SASRecDataset(args, user_seq, data_type="test")
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(
    #     test_dataset, sampler=test_sampler, batch_size=args.batch_size
    # )

    # model = S3RecModel(args=args)

    # trainer = FinetuneTrainer(
    #     model, train_dataloader, eval_dataloader, test_dataloader, None, args
    # )

    # print(args.using_pretrain)
    # if args.using_pretrain:
    #     pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
    #     try:
    #         trainer.load(pretrained_path)
    #         print(f"Load Checkpoint From {pretrained_path}!")

    #     except FileNotFoundError:
    #         print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    # else:
    #     print("Not using pretrained model. The Model is same as SASRec")

    # early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    # for epoch in range(args.epochs):
    #     trainer.train(epoch)

    #     scores, _ = trainer.valid(epoch)

    #     early_stopping(np.array(scores[-1:]), trainer.model)
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         break

    # trainer.args.train_matrix = test_rating_matrix
    # print("---------------Change to test_rating_matrix!-------------------")
    # # load the best model
    # trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    # scores, result_info = trainer.test(0)
    # print(result_info)


def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE



if __name__ == "__main__":
    main()
