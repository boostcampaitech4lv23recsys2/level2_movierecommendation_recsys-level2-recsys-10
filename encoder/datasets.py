import os
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset


class multiVAEDatasets(Dataset):
    def __init__(self, args, data, num_user, user_activity, data_type="train") -> None:
        super().__init__()
        self.args = args
        self.data = data
        # self.num_user = num_user
        self.user_activity = user_activity
        self.data_type = data_type
        # self.users = [i for i in range(num_user)]

    # def __len__(self):
    #     return self.num_user

    def __getitem__(self, idx):
        # user = self.users[idx]
        unique_uid = self.user_activity.index

        np.random.seed(98765)
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]

        n_users = unique_uid.size #31360
        n_heldout_users = 3000

        assert self.data_type in {"train", "valid", "test", "submission"}

        users = unique_uid[:(n_users - n_heldout_users * 2)]

        ##훈련 데이터에 해당하는 아이템들
        #Train에는 전체 데이터를 사용합니다.
        plays = self.data.loc[self.data['user'].isin(users)]
        ##아이템 ID
        unique_sid = pd.unique(plays['item'])
        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        pro_dir = os.path.join(self.args.data_dir, 'pro_sg')

        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)

        if self.data_type == "train":
            data_tr = self.numerize(plays, profile2id, show2id)
            data_tr.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
            data_te = data_tr
        else:
            if self.data_type == "valid":
                users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
            elif self.data_type == "test":
                users = unique_uid[(n_users - n_heldout_users):]
            else:
                users = []
            plays = self.data.loc[self.data['user'].isin(users)]
            plays = plays.loc[plays['item'].isin(unique_sid)]
            plays_tr, plays_te = self.split_train_test_proportion(plays)

            data_tr = self.numerize(plays_tr, profile2id, show2id)
            data_tr.to_csv(os.path.join(pro_dir, self.data_type+'_tr.csv'), index=False)

            data_te = self.numerize(plays_te, profile2id, show2id)
            data_te.to_csv(os.path.join(pro_dir, self.data_type+'_te.csv'), index=False)
                    
        print("Done!")
        return data_tr, data_te

    def get_train_valid_data(self):
        pass

    #훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
    #100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
    #확인하기 위함입니다.
    def split_train_test_proportion(data, test_prop=0.2):
        data_grouped_by_user = data.groupby('user')
        tr_list, te_list = list(), list()

        np.random.seed(98765)
        
        for _, group in data_grouped_by_user:
            n_items_u = len(group)
            
            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            
            else:
                tr_list.append(group)
        
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)

        return data_tr, data_te

    def numerize(tp, profile2id, show2id):
        uid = tp['user'].apply(lambda x: profile2id[x])
        sid = tp['item'].apply(lambda x: show2id[x])
        return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


class AEDataSet(Dataset):
    def __init__(self, num_user):
        self.num_user = num_user
        self.users = [i for i in range(num_user)]

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx): 
        user = self.users[idx]
        return torch.LongTensor([user])


class MatrixDataset(Dataset):
    def __init__(self, args, margs):
        self.args = args
        self.margs = margs
        self.df = pd.read_csv(os.path.join(self.args.data_dir, "train_ratings.csv"))

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder("item")
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder("user")
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df["item_idx"] = self.df["item"].apply(lambda x: self.item_encoder[x])
        self.df["user_idx"] = self.df["user"].apply(lambda x: self.user_encoder[x])

        self.user_train, self.user_valid = self.generate_sequence_data()

    def generate_encoder_decoder(self, col: str) -> dict:
        """
        encoder, decoder 생성
        Args:
            col (str): 생성할 columns 명
        Returns:
            dict: 생성된 user encoder, decoder
        """

        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def generate_sequence_data(self) -> dict:
        """
        sequence_data 생성
        Returns:
            dict: train user sequence / valid user sequence
        """
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        for user, item, time in zip(
            self.df["user_idx"], self.df["item_idx"], self.df["time"]
        ):
            users[user].append(item)

        for user in users:
            np.random.seed(self.args.seed)

            user_total = users[user]
            valid = np.random.choice(
                user_total, size=self.margs.valid_samples, replace=False
            ).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid  # valid_samples 개수 만큼 검증에 활용 (현재 Task와 가장 유사하게)

        return user_train, user_valid

    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    def make_matrix(self, user_list, train=True):
        """
        user_item_dict를 바탕으로 행렬 생성
        """
        mat = torch.zeros(size=(user_list.size(0), self.num_item))
        for idx, user in enumerate(user_list):
            if train:
                mat[idx, self.user_train[user.item()]] = 1
            else:
                mat[
                    idx, self.user_train[user.item()] + self.user_valid[user.item()]
                ] = 1
        return mat

    def make_sparse_matrix(self, test=False):
        X = sp.dok_matrix((self.num_user, self.num_item), dtype=np.float32)

        for user in self.user_train.keys():
            item_list = self.user_train[user]
            X[user, item_list] = 1.0

        if test:
            for user in self.user_valid.keys():
                item_list = self.user_valid[user]
                X[user, item_list] = 1.0

        return X.tocsr()