# :movie_camera: Movie Recommendation

## 1. 프로젝트 개요

### 1-1. 프로젝트 주제
![image_movie_recommendation_1](https://user-images.githubusercontent.com/79534756/206973144-f99f537b-2d5f-477e-9184-c35eacb8706b.JPG)
Competition 용도로 재구성된 MovieLens 데이터를 이용해 User의 영화 시청 이력 데이터를 바탕으로 User가 선호할 영화를 예측한다. User Sequence에서 일부 데이터가 누락된 상황을 가정했기 때문에, Timestamp를 고려한 User의 순차적인 이력과 Implicit Feedback을 함께 고려해야 하는 문제이다.

### 1-2. 프로젝트 기간
2022.12.12 ~ 2023.01.06(4주)

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

<br>

## 2. 프로젝트 팀 구성 및 역할
|[구혜인](https://github.com/hyein99?tab=repositories)|[권은채](https://github.com/dmscornjs)|[박건영](https://github.com/kuuneeee)|[장현우](https://github.com/jhu8802)|[정현호](https://github.com/Heiness)|[허유진](https://github.com/hobbang2)|
|----|----|----|----|----|----|
|MultiVAE/DAE 모델 구현 및 최적화|EASE모델 구현 및 최적화|SASRec, S3Rec모델 구현 및 최적화|FPMC 모델 구현 및 최적화|RecVAE 모델 구현 및 최적화|EDA, BERT4Rec 모델 구현 및 최적화|

<br>

## 3. 프로젝트 진행

### 3-1. 사전 기획
- 22.12.12(월) : Git branch 전략 회의
![Untitled](https://user-images.githubusercontent.com/49949138/215054280-ae1c99fc-212f-451c-880f-2e25469c1fab.png)
- 모델 탐색
    - 22.12.16(금) : 실습 기반 모델 세미나
    - 22.12.20(화) : 논문 기반 모델 세미나
- 베이스라인 코드 작성 및 실험 결과 공유
    - 22.12.23(금) : 베이스라인 세미나
      
### 3-2. 프로젝트 수행
![제목 없는 다이어그램 drawio](https://user-images.githubusercontent.com/49949138/215053398-cd6613a2-c352-4630-a69a-e4334805963c.png)
두 번의 세미나를 진행한 결과 Sequence 모델과 Encoder 모델이 MovieLens 데이터에 적절하다고 판단하였고, 2개의 세부 팀(Sequence팀, Encoder팀)으로 분리하여 프로젝트를 진행했다. Sequence팀, Encoder팀 각자 베이스라인 코드를 작성한 후 공유하는 세미나를 진행했다. 이후, 작성한 베이스라인 코드를 기준으로 테스트를 진행했다.

<br>

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
- 1. Sequence 계열 모델
	- 1) SASRec
	- 2) BERT4Rec
	- 3) FPMC
- 2. Encoder 계열 모델
	- 1) EASE
	- 2) MultiVAE/DAE
	- 3) RecVAE

### 4-3. 모델 선정
- 베이스라인 코드
    - SASRec & S3Rec
        - S3Rec의 사용 유무에 따른 성능 차이 실험
            - max_seq_len 과 hidden_dim 의 크기가 커질수록 유의미한 성능 상승이 있지만, GPU 허용량 문제 발생
                - SASRec만을 최적화했을 때 도출된 max_seq_len(448)과 hidden_dim(240)을 S3Rec pretrain시에 CUDA error 발생
                - S3Rec에서는 범위를 축소 max_seq_len (150~250) hidden_dim (30,60,120,240)해서 최적화
            - S3Rec에서는 상대적으로 짧은 max_seq_len에서 동일한 Recall@10 결과가 나왔으므로 의미가 있으나 많은 연산량과 높은 소요 시간 때문에 S3Rec만 최적화하는 것으로 결정
            - S3Rec Pretrain 사용(sweep 적용) Recall@10 : 0.1294
            - S3Rec Pretrain 미사용(sweep 적용) Recall@10 : 0.1294
            - 연산 소요 시간을 비교
                - S3Rec 사용시 : 약 13시간
                - S3Rec 미사용시 : 약 3시간
- 추가적인 모델 선택
    - Sequence 모델
        - FPMC
            - Movie Recommendation는 User의 Implicit Feedback을 사용하여 다음 Item을 추천하는 프로젝트이기 때문에 MF 모델에 Markov Chains를 적용한 FPMC 모델이 적절하다고 판단
        - BERT4Rec
            - BERT4Rec 에서 사용하는 Cloze mask 방식이 개요에서 소개된 시청 이력 누락과 유사하다고 판단
    - Encoder 모델
        - Movie Recommendation은 Top-K Ranking을 사용하는 프로젝트이기 때문에 Encoder, Decoder를 사용하여 예측하는 Encoder 모델이 적절하다고 판단, Static 데이터를 사용한 추천이기 때문에 Encoder 모델 중 MultiVAE, MultiDAE, RecVAE, EASE 사용
	
### 4-4. 모델 성능 개선 방법
- Hyperparameter Tuning(Wandb, Sweep)
    - Sweep
- Ensemble
    - Top-K Counting
    	- Ensemble대상 모델의 Recall@20/15/10을 추출하여 추천 Item Count
    	- 추천 순위 별 가중치, 모델 별 가중치를 적용하여 테스트를 진행
    - 모델 별 상위 N개 추출
    	- Ensemble 대상 모델 중 상위 N개를 추출하여 10개의 추천 Item으로 구성
    	- N개의 기준은 모델의 성능 별 가중치 부여

<br>

## 5. WrapUp Report
[Level_2_MovieRecommendation_랩업리포트](https://www.notion.so/Level_2_MovieRecommendation_-c55d747e6dfb408ea7c378cba5576818)

