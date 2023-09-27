Neural Collavorative Filtering
====================================

### 개요
- Fusion of GMF and MLP
$$\phi^{GMF}=P\odot Q$$

$$\phi^{MLP}=a_L(a_{L-1}((\cdots a_2(\begin{pmatrix}
P & Q \\
\end{pmatrix}W^T_2+b_2)\cdots)W_{L-1}^T+b_{L-1})W_L^T+b_L)$$

$$Y=a_{out}(\begin{pmatrix}
\phi^{GMF} & \phi^{MLP} \\
\end{pmatrix}h)$$

GMF는 두 Latent 행렬 간의 선형적인 관계를 모델링 하고 MLP는 비선형적 관계를 모델링한다.
이 둘을 조합한 모형을 Neural Matrix Factorization(NeuMF)라고 하며 선형적 관계와 비선형적 관계를 모두 잘 표현할 수 있는 모형을 목표로 한다.

### 데이터 설명
- 분석에 사용한 데이터는 MovieLens-1m 데이터로 6,040명의 사용자가 3,706개의 영화를 인당 최소 20편 이상 점수를 매긴 자료이다. 총 1,000,209개의 점수가 기록되어있다.
점수는 1~5점의 값을 가지지만 implicit signal 분석을 위해 사용자가 영화의 점수를 기록하면 1 아니면 0으로 변환하였다. 

|Dataset|Interaction#|Item#|User#|Sparsity|
|----|------|------|-----|-----|
|MovieLens-1m|1,000,209|3,706|6,040|95.53%|

- Negative Sampling
상호작용이 있는 경우는 긍정적 상호작용으로 간주하여 모두 학습에 포함시켰고 상호작용이 없는 경우는 일부를 추출하여 부정적 상호작용으로 간주하였다. 긍정적 상호작용 1회당 부정적 상호작용 4회를 추출하였다.

### 분석
- Evaluation
평가를 위한 Test Dataset은 각 사용자마다 마지막 상호작용 1회와 무작위로 추출한 부정적 상호작용 99회로 이루어져있다. 각 사용자마다 100회의 상호작용에 대한 예측확률을 구한 뒤 상위에 실제 상호작용(마지막 상호작용)이 있는지 확인하는 방식으로 모형을 평가하였다.
모형의 평가지표는 HitRatio@10와 NDCG@10를 사용하였다. 

- Hyper Parameter
Batch size : 512
Learning rate : 0.001
Predictive factors : 8
$$\alpha$$ : 0.5
Embedding size : 16

### 결과
![loss.png](https://www.dropbox.com/scl/fi/ly2d0q7uzrlkwrdj6taam/loss.png?rlkey=ker770d3w89b5p80dzeerttbj&dl=0&raw=1)
![hr.png](https://www.dropbox.com/scl/fi/cmlu4xgiob7duwigukad2/hr.png?rlkey=tfijdsx99zwn1wu4femp6vj0e&dl=0&raw=1)
![ndcg.png](https://www.dropbox.com/scl/fi/pnhyvhxkszbyz8c89h85s/ndcg.png?rlkey=kby4g0eeuutibg47c527hocvc&dl=0&raw=1)
Pretrained NeuMF > GMF > MeuMF > MLP 순으로 높은 HitRatio@10와 NDCG@10 값을 보였다.
