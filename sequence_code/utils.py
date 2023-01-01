import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set:set, item_size:int):
    """1부터 item_size - 1 까지의 값에서 제공된 item_set 에 없는 값을 반환한다. 

    Args:
        item_set (set): int 값을 element 로 가지는 set
        item_size (int): randomint 를 추출할 최대 값

    Returns:
        _type_: 주어진 
    """
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(data_file, preds):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/submission.csv", index=False
    )


def generate_submission_file_v2(data_file, preds, item2idx_):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item2idx_[item]))  # title index화한거 item으로 되돌리기

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/submission.csv", index=False
    )


def get_user_seqs(data_file, item2idx_, random_sort=0 , b_sort_by_time:bool=False):
    
    item2idx_, idx2item_ = indexinfo.get_index_info()
    rating_df = pd.read_csv(data_file)
    rating_df['item'] = rating_df['item'].map(lambda x: item2idx_[x])

    if( True == b_sort_by_time ) :
        rating_df.sort_values(['user', 'time'], inplace=True) 

    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    item_set = set()
    for line in lines:

        items = line
        if random.random() < random_sort:
            random.shuffle(items)
        user_seq.append(items)
        item_set = item_set | set(items)
        
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    submission_rating_matrix = generate_rating_matrix_submission(
        user_seq, num_users, num_items
    )
    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(data_file:str,  item2idx_:dict, random_sort=0):
    """user가 본 item 기록이 있는 data file 을 받아서 sequence 정보를 반환한다. 

    Args:
        data_file (str): user가 본 item 목록이 있는 data file ( train_ratings.csv )

    Returns:
        list: user 별로 본 영화의 id list, [[],..] 
        int : 전체 user 가 본 movie 의 unique id 개수, 
        list: 전체 user 가 본 movie list ( 중복 O ) []
    """
    rating_df = pd.read_csv(data_file)
    rating_df['item'] = rating_df['item'].map(lambda x: item2idx_[x]) # 모든 item을 title index로 변환
    # user id 로 group 하여 item 목록 추출 : pandas seriese
    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    long_sequence = []
    item_set = set()
    # 한 명의 user 가 본 item list 순회
    for line in lines:
        items = line
        if random.random() < random_sort:   # 시청한 기록 shuffle / random_sort 기본값 0 / random.random() -> 0<=v<1
            random.shuffle(items)
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


def get_item2attribute_json(data_file:str):
    """_summary_

    Args:
        data_file (str): 전처리에 의해 생성된 데이터(item과 genre의 mapping 데이터)의 file name

    Returns:
        list: item 과 attribute 가 mapping 된 객체가 담긴 list 
        int : movie attribute 의 unique 한 개수 
    """
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def get_popular_items(data_file, item2idx_, p):
    rating_df = pd.read_csv(data_file)
    rating_df['item'] = rating_df['item'].map(lambda x: item2idx_[x])
    popular_items = list(rating_df["item"].value_counts().index)
    return popular_items[:int(len(popular_items)*p)+1]

def neg_sample_from_popular_items(item_set, popular_items, max_len):
    sample = random.choice(popular_items)
    while sample in item_set:
        sample = random.choice(popular_items)
    return sample

def generate_item2idx():
    item2idx = pd.read_csv('../data/train/item2idx.tsv', sep='\t', index_col=0, names=['item_id'])
    idx2item = pd.read_csv('../data/train/item2idx.tsv', sep='\t', index_col=1, names=['item'])
    item2idx_ = dict()
    idx2item_ = dict()
    for x in item2idx['item_id'].index[1:]:
        item2idx_[int(x)] = item2idx['item_id'][x]
    for i in idx2item['item'].index[1:]:
        idx2item_[i] = int(idx2item['item'][i])
    return item2idx_, idx2item_


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class IndexInfo : 

    def __init__(self, data_path: str ) : 

        item2idx_ = dict()
        idx2item_ = dict()
    
        self.item2idx_, self.idx2item_ = self.make_idx_mapping_info( data_path, item2idx_,idx2item_)

    def make_idx_mapping_info(self,data_path:str,item2idx_:dict, idx2item_:dict):

        if( True == os.path.isfile(data_path)): 
            item2idx = pd.read_csv(data_path, sep='\t', index_col=0, names=['item_id'])
            idx2item = pd.read_csv(data_path, sep='\t', index_col=1, names=['item'])
            
            for x in item2idx['item_id'].index[1:]:
                item2idx_[int(x)] = item2idx['item_id'][x]
            for i in idx2item['item'].index[1:]:
                idx2item_[i] = int(idx2item['item'][i])
        
        return item2idx_, idx2item_

    # data_path 를 생성하는 preprocessing.py 에서 item2idx_ 를 사용하기 때문에 new path 설정 추가 
    def get_index_info(self, new_path:str=""):

        if( True == os.path.isfile(new_path)): 
            self.item2idx_, self.idx2item_ = self.make_idx_mapping_info(new_path, self.item2idx_, self.idx2item_)

        if (self.item2idx_) and (self.idx2item_):
            return self.item2idx_, self.idx2item_

        raise Exception("Please check about datapath. The data will be genereated in pretrain process ( run_pretrain.py )")

indexinfo = IndexInfo('../data/train/item2idx.tsv')   
item2idx_,idx2item_=indexinfo.get_index_info()