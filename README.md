# Movie Recommendation

`Movie Recommendation`  
- 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 **다음에 시청할 영화 및 좋아할 영화를 예측**

![image](https://s3-us-west-2.amazonaws.com/aistages-prod-server-public/app/Users/00000068/files/b147dabe-613e-4ebf-b605-b615f032608d..png)

***기간*** : 2022.12.12 ~ 2022.1.06(4주)



***Description*** :

> **GOAL** : **사용자가 다음에 시청할 영화 및 좋아할 영화를 예측**

> 데이터 설명
- `train_ratings.csv` : 주 학습 데이터, userid, itemid, timestamp(초)로 구성 - 5,154,471 행

- `Ml_item2attributes.json` : 전처리에 의해 생성된 데이터(item과 genre의 mapping 데이터)

- `titles.tsv` : 영화 제목 - 6,807행

- `years.tsv` : 영화 개봉년도 - 6,799행

- `directors.tsv` : 영화별 감독 - 5,905 행

- `genres.tsv` : 영화 장르 (한 영화에 여러 장르가 포함될 수 있음) - 15,934 행

- `writers.tsv` : 영화 작가 - 11,307행

		
- ***Submission*** : 

	- Submission 파일 포맷

	- csv 형태

	- 훈련 데이터의 있는 전체 유저에 대해 각각 10개씩 추천

	- header제외 313,600 행

- ***Metric*** : 
	![image](https://s3-us-west-2.amazonaws.com/aistages-prod-server-public/app/Users/00000068/files/6ea4c101-a327-45bb-8f1c-08307208ea3a..png)

	- (normalized) Recall@10

	- 강의에서 배운 Recall@K와 분모 부분에서 약간 차이를 보이며, (normalized) Recall@K는 `min(K,|I_u|)`를 분모로 사용함으로써 K와 사용자가 사용한 아이템 수 `|I_u|` 중 최소값을 분모로 사용한다.

	- 이것은 상위 K개에 위치한 관련된 아이템들을 ranking함으로써, Recall@K 값을 최대 1로 normalize하는 효과를 가진다. 참고: https://arxiv.org/pdf/1802.05814.pdf

	- `Recall 값이 클 수록 높은 순위`이며, 만약 값이 같다면 `제출 횟수가 적은 팀`이 더 높은 등수를 갖게 된다.

	


## 📁프로젝트 구조

```
├── code
│   ├── datasets.py
│   ├── inference.py
│   ├── models.py
│   ├── modules.py
│   ├── output
│   │   └── most_popular_submission.csv
│   ├── preprocessing.py
│   ├── requirements.txt
│   ├── run_pretrain.py
│   ├── run_train.py
│   ├── sample_submission.ipynb
│   ├── trainers.py
│   └── utils.py
```

## :man_technologist: Members
구혜인 권은채 박건영 장현우 정현호 허유진


