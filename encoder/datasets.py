import pandas as pd
import numpy as np
import scipy.sparse as sp

import torch.nn as nn
from torch.utils.data import Dataset

import os
from time import time
from tqdm import tqdm

class MultiVAEDataset(Dataset):
    def __init__(self, args):
        self.path = args.data_dir

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.exist_users = []

        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        data_path = os.path.join(self.path, args.train_data)
        df = pd.read_csv(data_path)

        item_ids = df['item'].unique() # 아이템 고유 번호 리스트
        user_ids = df['user'].unique() # 유저 고유 번호 리스트
        self.n_items, self.n_users = len(item_ids), len(user_ids)
        
        # user, item indexing
        item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids) 
        user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) 

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        del df['item'], df['user']

        self.exist_items = list(df['item_idx'].unique())
        self.exist_users = list(df['user_idx'].unique())

        t1 = time()
        self.train_items, self.valid_items = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list)
        
        print('Creating interaction Train/ Vaild Split...')
        for uid, item in enumerate(items):            
            num_u_valid_items = min(int(len(item)*0.125), 10) # 유저가 소비한 아이템의 12.5%, 그리고 최대 10개의 데이터셋을 무작위로 Validation Set으로 활용한다.
            u_valid_items = np.random.choice(item, size=num_u_valid_items, replace=False)
            self.valid_items[uid] = u_valid_items
            self.train_items[uid] = list(set(item) - set(u_valid_items))

        self.train_data = pd.concat({k: pd.Series(v) for k, v in self.train_items.items()}).reset_index(0)
        self.train_data.columns = ['user', 'item']

        self.valid_data = pd.concat({k: pd.Series(v) for k, v in self.valid_items.items()}).reset_index(0)
        self.valid_data.columns = ['user', 'item']

        print('Train/Vaild Split Complete. Takes in', time() - t1, 'sec')
        
        rows, cols = self.train_data['user'], self.train_data['item']
        self.train_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))
        self.train_input_data = self.train_input_data.toarray()

        # bm25_weight
        # self.train_input_data = bm25_weight(self.train_input_data, K1=100, B=0.9)
        # values = self.train_input_data.data
        # indices = np.vstack((self.train_input_data.row, self.train_input_data.col))

        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = self.train_input_data.shape

        # self.train_input_data = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_input_data[idx,:]


class MultiVAEValidDataset(Dataset):
    def __init__(self, train_dataset):
        self.n_users = train_dataset.n_users
        self.n_items = train_dataset.n_items
        self.train_input_data = train_dataset.train_input_data

        
        self.valid_data = train_dataset.valid_data
        rows, cols = self.valid_data['user'], self.valid_data['item']
        self.valid_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))

        self.valid_input_data = self.valid_input_data.toarray()
    
    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_input_data[idx, :], self.valid_input_data[idx,:]


class RecVAEDataset(Dataset):
    def __init__(self, args, mode = 'train'):
        self.path = args.data_dir

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        data_path = os.path.join(self.path, args.train_data)
        genre_path = os.path.join(self.path, 'genres.tsv')
        df = pd.read_csv(data_path)
        genre_data = pd.read_csv(genre_path, sep='\t')

        ############### item based outlier ###############
        # # 아이템 기준 outlier 제거 - 이용율 0.3% 미만인 아이템 제거 (영구히 제거)
        # item_freq_df = (df.groupby('item')['user'].count()/df.user.nunique()).reset_index()
        # item_freq_df.columns = ['item', 'item_freq']
        # # df = df.merge(item_freq_df, on='item').query('item_freq > 0.003')
        # # df = df.merge(item_freq_df, on='item').query('item_freq > 0.005')
        # df = df.merge(item_freq_df, on='item').query('item_freq > 0.01')
        # del df['item_freq'] # 소명을 다하고 삭제! 

        self.ratings_df = df.copy() # for submission
        self.n_train = len(df)

        item_ids = df['item'].unique() # 아이템 고유 번호 리스트
        user_ids = df['user'].unique() # 유저 고유 번호 리스트
        self.n_items, self.n_users = len(item_ids), len(user_ids)
        
        # user, item indexing
        # item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item) ; 아이템을 1부터 설정하는이유? 0을 아무것도 아닌 것으로 blank 하기 위해서
        self.item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids) # item re-indexing (0~num_item-1) ; 아이템을 1부터 설정하는이유? 0을 아무것도 아닌 것으로 blank 하기 위해서
        self.user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': self.item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': self.user2idx[user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        genre_data = df.merge(genre_data, on = 'item').copy()
        del df['item'], df['user']

        self.exist_items = list(df['item_idx'].unique())
        self.exist_users = list(df['user_idx'].unique())

        ############### Used by Sampler ###############
        # # 1. user-based outlier - 상위 20퍼센트 영화를 본 친구들 Weight=0 지정
        # self.user_weights = np.ones_like(self.exist_users)
        # outlier_users = df['user_idx'].unique()[df.groupby('user_idx').item_idx.count()/df['item_idx'].nunique() >= 0.4]
        # self.user_weights[outlier_users] = 0

        t1 = time()
        self.train_items, self.valid_items = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list) # 유저 아이디 상관 없이, 순서대로 
        if mode == 'train':
            print('Creating interaction Train/ Vaild Split...')
            for uid, item in enumerate(items):            
                num_u_valid_items = min(int(len(item)*0.125), 10) # 유저가 소비한 아이템의 12.5%, 그리고 최대 10개의 데이터셋을 무작위로 Validation Set으로 활용한다.
                ####### Original method : RANDOM #######
                # u_valid_items = np.random.choice(item, size=num_u_valid_items, replace=False)
                # self.valid_items[uid] = u_valid_items
                # self.train_items[uid] = list(set(item) - set(u_valid_items))

                ####### method-1 : Last sequence ####### 마지막 sequence에 있는 정보를 제거
                # u_valid_items = item[-num_u_valid_items:]
                # self.valid_items[uid] = u_valid_items
                # self.train_items[uid] = list(set(item) - set(u_valid_items))

                ####### method-2 : hybrid ####### 마지막꺼:무작위= 1:1
                # num_random = int(num_u_valid_items//2 + num_u_valid_items%2) # 홀수일때는, 무작위로 뽑는것이 1개 더 많게
                # num_last = int(num_u_valid_items - num_random)
                # last_items = item[-num_last:]
                # random_items = np.random.choice(item[:-num_last], size=num_random, replace=False).tolist()
                # u_valid_items = random_items + last_items
                # self.valid_items[uid] = u_valid_items
                # self.train_items[uid] = list(set(item) - set(u_valid_items))

                ####### method-3 : hybrid ####### 마지막꺼:무작위= 6:4
                num_random = np.floor(num_u_valid_items*0.6).astype(int) # 홀수일때는, 무작위로 뽑는것이 1개 더 많게
                num_last = int(num_u_valid_items - num_random)
                last_items = item[-num_last:]
                random_items = np.random.choice(item[:-num_last], size=num_random, replace=False).tolist()
                u_valid_items = random_items + last_items
                self.valid_items[uid] = u_valid_items
                self.train_items[uid] = list(set(item) - set(u_valid_items))

            self.train_data = pd.concat({k: pd.Series(v) for k, v in self.train_items.items()}).reset_index(0)
            self.train_data.columns = ['user', 'item']

            self.valid_data = pd.concat({k: pd.Series(v) for k, v in self.valid_items.items()}).reset_index(0)
            self.valid_data.columns = ['user', 'item']
        
        if mode == 'train_all': #else
            print('Preparing interaction all train set')
            # for uid, item in enumerate(items):            
            #     self.train_items[uid] = item

            # self.train_data = pd.concat({k: pd.Series(v) for k, v in train_items.items()})
            # self.train_data.reset_index(0, inplace=True)
            # self.train_data.columns = ['user', 'item']
            self.train_data = pd.DataFrame()
            self.train_data['user'] = df['user_idx']
            self.train_data['item'] = df['item_idx']

        print('Train/Vaild Split Complete. Takes in', time() - t1, 'sec')
        
        rows, cols = self.train_data['user'], self.train_data['item']
        self.train_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))
        self.train_input_data = self.train_input_data.toarray()

        print('Making Genre filter ... ')
        genre2item = genre_data.groupby('genre')['item_idx'].apply(set).apply(list)
        # user2genre = genre_data.groupby('user_idx')['genre'].apply(set).apply(list)

        genre_data_freq = genre_data.groupby('user_idx')['genre'].value_counts(normalize=True)
        genre_data_freq_over_5p = genre_data_freq[genre_data_freq > 0.005].reset_index('user_idx')
        genre_data_freq_over_5p.columns = ['user_idx', 'tobedroped']
        genre_data_freq_over_5p = genre_data_freq_over_5p.drop('tobedroped', axis = 1).reset_index()
        user2genre = genre_data_freq_over_5p.groupby('user_idx')['genre'].apply(set).apply(list)

        genre2item_dict = genre2item.to_dict()
        all_set_genre = set(genre_data['genre'].unique())
        user_genre_filter_dict = {}
        for user, genres in tqdm(enumerate(user2genre)):
            unseen_genres = all_set_genre - set(genres) # set
            unseen_genres_item = set(sum([genre2item_dict[genre] for genre in unseen_genres], []))
            user_genre_filter_dict[user] = pd.Series(list(unseen_genres_item), dtype=np.int32)

        user_genre_filter_df = pd.concat(user_genre_filter_dict).reset_index(0)
        user_genre_filter_df.columns = ['user', 'item']
        user_genre_filter_df.index = range(len(user_genre_filter_df))

        rows, cols = user_genre_filter_df['user'], user_genre_filter_df['item']
        self.user_genre_filter = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))
        self.user_genre_filter = self.user_genre_filter.toarray()


    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_input_data[idx,:]

class RecVAEValidDataset(Dataset):
    def __init__(self, train_dataset, genre_filter = False):
        self.n_users = train_dataset.n_users
        self.n_items = train_dataset.n_items
        self.train_input_data = train_dataset.train_input_data
        self.user_genre_filter = train_dataset.user_genre_filter
        self.genre_filter = genre_filter

    
        self.valid_data = train_dataset.valid_data
        rows, cols = self.valid_data['user'], self.valid_data['item']
        self.valid_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))

        self.valid_input_data = self.valid_input_data.toarray()
    
    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        input_data = self.train_input_data[idx, :]
        label_data = self.valid_input_data[idx,:]
        if self.genre_filter:
            genre_filter = (1-self.user_genre_filter[idx,:]) > 0
            label_data =  np.logical_and(label_data, genre_filter).astype(np.float32)
        return input_data, label_data

class BeforeNoiseUnderSamplingDataset(RecVAEDataset):
    def __init__(self, path='../data/', mode='train'):
        super().__init__(path, mode)

    # def noise_without_pos(self, u, num):
    #     pos_items = self.train_input_data[u]
    #     # n_pos_items = len(pos_items)
    #     pos_batch = []
    #     while True:
    #         if len(pos_batch) == num: break
    #         pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
    #         pos_i_id = pos_items[pos_id]

    #         if pos_i_id not in pos_batch:
    #             pos_batch.append(pos_i_id)
    #     return pos_batch


    def __getitem__(self, idx):
        # noise = np.random.choice(2, size=[*self.train_input_data.shape], p=[0.9, 0.1])
        # train_input_data_noised = self.train_input_data - noise
        # train_input_data_noised[train_input_data_noised < 0] = 0
        # return train_input_data_noised[idx,:]
        # noise = np.random.choice(2, size=[*self.train_input_data.shape], p=[0.9, 0.1])
        # noise = np.random.choice(2, size= len(self.train_input_data[idx,:]),  p=[0.9, 0.1]).astype(np.float32)
        # train_input_data_noised = self.train_input_data + noise
        # train_input_data_noised[train_input_data_noised < 0] = 0
        return self.train_input_data[idx,:], np.random.randint(0,2,size=self.train_input_data.shape[1]).astype(np.float32)