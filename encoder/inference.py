import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from preprocessing import numerize

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--train_data", default="train_ratings.csv", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="multiVAE", type=str)

    # train args
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
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
     
    
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )

    parser.add_argument('--save', type=str, default='output/model.pt',
                        help='path to save the final model')

    args = parser.parse_args()

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    args.device = torch.device("cuda" if args.cuda else "cpu")

    model_dir = os.path.join(args.output_dir, args.save)

    # Load the best saved model.
    with open(model_dir, 'rb') as f:
        model = torch.load(f)

    rating_df = pd.read_csv(os.path.join(args.data_dir, args.train_data), header=0)

    test_unique_uid = pd.unique(rating_df['user'])
    test_unique_sid = pd.unique(rating_df['item'])

    n_items = len(pd.unique(rating_df['item']))

    show2id = dict((sid, i) for (i, sid) in enumerate(test_unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(test_unique_uid))

    test_rating_df = numerize(rating_df, profile2id, show2id)

    n_users = test_rating_df['uid'].max() + 1
    rows, cols = test_rating_df['uid'], test_rating_df['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                            (rows, cols)), dtype='float64',
                            shape=(n_users, n_items))

    test_data_tensor = torch.FloatTensor(data.toarray()).to(args.device)

    recon_batch, mu, logvar = model(test_data_tensor)

    id2show = dict(zip(show2id.values(),show2id.keys()))
    id2profile = dict(zip(profile2id.values(),profile2id.keys()))

    result = []

    for user in range(len(recon_batch)):
        rating_pred = recon_batch[user]
        rating_pred[test_data_tensor[user].reshape(-1) > 0] = 0

        idx = np.argsort(rating_pred.detach().cpu().numpy())[-10:][::-1]
        for i in idx:
            result.append((id2profile[user], id2show[i]))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/" + args.model_name + ".csv", index=False
    )


    

if __name__ == "__main__":
    main()
