import pandas as pd

""" model 학습 전에 수행되어 전처리된 json file 을 생성 
"""

def main():
    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array
    genres_df.groupby("item")["genre"].apply(list).to_json(
        "data/Ml_item2attributes.json"
    )


if __name__ == "__main__":
    main()
