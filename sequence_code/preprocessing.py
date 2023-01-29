import pandas as pd
import numpy as np
from utils import indexinfo

""" model 학습 전에 수행되어 전처리된 json file 을 생성 
"""

def main():
    item_df = pd.read_csv('../data/train/titles.tsv', sep='\t')
    item_ids = item_df['item'].unique()
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids)
    # index -> item / value -> item_index(새로 지정 1-6807) : item 번호를 idx화 해서 사용하겠다는 의미
    item2idx.to_csv('../data/train/item2idx.tsv', sep='\t', encoding='utf-8', index=True)

    item2idx_,_ = indexinfo.get_index_info(new_path = '../data/train/item2idx.tsv')

    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    genres_df["item"] = genres_df['item'].map(lambda x: item2idx_[x])
    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array
    genres_df.groupby("item")["genre"].apply(list).to_json(
        "../data/train/Ml_item2attributes.json"
    )

if __name__ == "__main__":
    main()
