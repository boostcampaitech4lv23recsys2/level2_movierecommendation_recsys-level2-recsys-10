import os
import pandas as pd
from scipy import sparse
import numpy as np

def preprocessing(data):
    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(data, min_uc=5, min_sc=0)
    #제공된 훈련데이터의 유저는 모두 5개 이상의 리뷰가 있습니다.
    # print("5번 이상의 리뷰가 있는 유저들로만 구성된 데이터\n",raw_data)
    # print("유저별 리뷰수\n",user_activity)
    # print("아이템별 리뷰수\n",item_popularity)
    return user_activity


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count


# 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상) 
# 데이터만을 추출할 때 사용하는 함수입니다.
# 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount



