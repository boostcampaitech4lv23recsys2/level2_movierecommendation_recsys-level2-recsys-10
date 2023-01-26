# :movie_camera: Movie Recommendation

## 1. 프로젝트 개요
### 1-1. 프로젝트 주제
![image_movie_recommendation_1](https://user-images.githubusercontent.com/79534756/206973144-f99f537b-2d5f-477e-9184-c35eacb8706b.JPG)
Competition 용도로 재구성된 MovieLens 데이터를 이용해 User의 영화 시청 이력 데이터를 바탕으로 User가 선호할 영화를 예측한다. User Sequence에서 일부 데이터가 누락된 상황을 가정했기 때문에, Timestamp를 고려한 User의 순차적인 이력과 Implicit Feedback을 함께 고려해야 하는 문제이다.
### 1-2. 프로젝트 기간
2022.12.12 ~ 2022.1.06(4주)
### 1-3. 활용 장비 및 재료
- 개발환경 : VScode, PyTorch, Jupyter, Ubuntu 18.04.5 LTS, GPU Tesla V100-PCIE-32GB
- 협업 Tool : GitHub, Notion
- 시각화 : WandB
### 1-4. 프로젝트 구조도
- (1) Sequence folder
    - BERT4Rec
    - FPMC
    - SASRec
    - S3Rec
- (2) Encoder folder
    - EASE
    - MultiDAE
    - MultiVAE
    - RecVAE
- (3) Ensemble
- (4) EDA
### 1-5. 데이터 구조
```
train
├── Ml_item2attributes.json
├── directors.tsv
├── genres.tsv
├── titles.tsv
├── train_ratings.csv
├── writers.tsv
└── years.tsv
```

## 2. 프로젝트 팀 구성 및 역할
|[구혜인](https://github.com/hyein99?tab=repositories)|[권은채](https://github.com/dmscornjs)|[박건영](https://github.com/kuuneeee)|[장현우](https://github.com/jhu8802)|[정현호](https://github.com/Heiness)|[허유진](https://github.com/hobbang2)|
|----|----|----|----|----|----|
|MultiVAE/DAE 모델 구현 및 최적화|EASE모델 구현 및 최적화|SASRec, S3Rec모델 구현 및 최적화|FPMC 모델 구현 및 최적화|RecVAE 모델 구현 및 최적화|EDA, BERT4Rec 모델 구현 및 최적화|

## 3. 프로젝트 진행
### 3-1. 사전 기획
### 3-2. 프로젝트 수행

## 4. 프로젝트 수행 결과
### 4-1. 모델 성능 및 결과
**■ 결과 (상위 4 개) : Publie, Private 4위 🏅**
| SASRec | BERT | FPMC | EASE | multiVAE | multiDAE | RecVAE |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1294 | 0.0479 | 0.1278 | 0.1600 | 0.1356 | 0.1376 | 0.1505 |

| 최종 선택 여부 | 모델 (Ensemble 비율) | Public Recall@10 | Private Recall@10 |
| --- | --- | --- | --- |
| X | EASE 와 SASRec 을 7:3 비율로 섞음  | 0.1755 | 0.1655 |
| O | EASE (1), RecVAE(0.9), MultiDAE(0.8), MultiVAE(0.7), Sasrec(1) | 0.1726 | 0.1651 |
| X | EASE , RecVAE, MultiDAE, MultiVAE, SASRec, Recall@10 순위, 모델 가중치 | 0.1630 | 0.1623 |
| O | EASE 와 SASRec 을 5:5 비율로 섞음  | 0.1758 | 0.1615 |
### 4-2. 모델 개요
### 4-3. 모델 선정
### 4-4. 모델 성능 개선 방법

## 5. WrapUp Report
[Level_2_MovieRecommendation_랩업리포트](https://www.notion.so/Level_2_MovieRecommendation_-c55d747e6dfb408ea7c378cba5576818)




<br><br>

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
	![image_movie_recommendation_2](https://user-images.githubusercontent.com/79534756/206973347-6ca2ab45-8a0c-4602-ba0f-80e8d04dd8a0.JPG)

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


