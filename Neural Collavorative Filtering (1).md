Neural Collavorative Filtering Review
===================================
### Collavorative Filtering
- Notation
$$M:\#\;of\;user$$
$$N:\#\;of\;item$$
$$K:Latent\;dimension\;number$$
$$Y\in\mathbb{R}^{M\times N}$$
$$y_{ui}=\begin{cases}
 1,\; if \;interaction \;is \;observed\\0, \;otherwise.
\end{cases}.$$
$$P\in\mathbb{R}^{M\times K}$$
$$Q\in\mathbb{R}^{N\times K}$$

- Matrix Factorization
$$Y=PQ^T$$
collavorative filtering은 interaction 행렬인 Y를 두 행렬의 곱으로 분해하는 것이다.
이 논문에서는 Y를 행렬곱셈이 아닌 좀더 복잡한 연산으로 분해함으로서 더 유연한 예측을 목표로 한다.

### Generalized Matrix Factorization
- $$Y=a_{out}((P\odot Q)h)$$
$$\odot :Hadamard\; product( element-wise \;product)$$
$$a_{out}:output\;function$$
위의 행렬 분해를 확장한 것으로 H행렬이 1행렬이고 output함수가 상수함수이면 기존 행렬 분해와 동일하다. 
이 논문에서 output 함수는 sigmoid 함수를 사용하였다.

### Multi-Layer Perceptron
- $$Z_1=\begin{pmatrix}
P & Q \\
\end{pmatrix}$$
$$Z_2=a_2(Z_1W^T_2+b_2)$$
$$\vdots$$
$$Z_L=a_L(Z_{L-1}W_L^T+b_L)$$
$$Y=a_{out}(Z_Lh)$$
$$a_i:ReLU,\;i=1,2,\cdots,L$$
다층 신경망 모형을 통해 Y와 두 행렬 간의 관계를 모델링 한다.
이 논문에서는 총 3개의 hidden layer를 사용하였으며 layer의 차원은 절반씩 줄어들며 마지막 layer의 차원을 Predictive factors라고 한다.

### Fusion of GMF and MLP
- $$\phi^{GMF}=P\odot Q$$
$$\phi^{MLP}=a_L(a_{L-1}((\cdots a_2(\begin{pmatrix}
P & Q \\
\end{pmatrix}W^T_2+b_2)\cdots)W_{L-1}^T+b_{L-1})W_L^T+b_L)$$
$$Y=a_{out}(\begin{pmatrix}
\phi^{GMF} & \phi^{MLP} \\
\end{pmatrix}h)$$
GMF는 두 Latent 행렬 간의 선형적인 관계를 모델링 하고 MLP는 비선형적 관계를 모델링한다.
이 둘을 조합한 모형을 Neural Matrix Factorization(NeuMF)라고 하며 선형적 관계와 비선형적 관계를 모두 잘 표현할 수 있는 모형을 목표로 한다.
![화면 캡처 2023-09-21 182932.png](https://www.dropbox.com/scl/fi/y851l93qeo92lh4eoixoq/2023-09-21-182932.png?rlkey=0acs2rnytxpot2i06ko5l7yqj&dl=0&raw=1)

### Pretrain
- NeuMF는 GMF와 MLP의 조합으로 구성된다. 따라서 각각을 학습시킨 parameter를 NeuMF의 parameter 초기값으로 사용할 수 있다. 이때 ouput layer의 parameter는 두 모형의 output layer parameter의 가중평균으로 한다. 
$$h\leftarrow \begin{pmatrix}
\alpha h^{GMF} \\ (1-\alpha)h^{MLP}
\end{pmatrix}$$

- GMF와 MLP를 학습할 때는 parameter 초기값으로 평균이 0이고 표준편차가 0.01인 정규분포에서 무작위 추출하여 사용한다.
Optimizer는 기본적으로 Adam을 사용하지만 Pretrain을 한 NeuMF의 경우 vanilla SGD를 사용한다.

### 데이터 설명
- 분석에 사용한 데이터는 MovieLens-1m 데이터로 6,040명의 사용자가 3,706개의 영화를 인당 최소 20편 이상 점수를 매긴 자료이다. 총 1,000,209개의 점수가 기록되어있다.
점수는 1~5점의 값을 가지지만 implicit signal 분석을 위해 사용자가 영화의 점수를 기록하면 1 아니면 0으로 변환하였다. 

|Dataset|Interaction#|Item#|User#|Sparsity|
|----|------|------|-----|-----|
|MovieLens-1m|1,000,209|3,706|6,040|95.53%|

- Negative Sampling
상호작용이 있는 경우는 긍정적 상호작용으로 간주하여 모두 학습에 포함시켰고 상호작용이 없는 경우는 일부를 추출하여 부정적 상호작용으로 간주하였다. 긍정적 상호작용 1회당 부정적 상호작용 4회를 추출하였다.

#### 분석
- Loss Function
Sigmoid 함수를 통해 사용자가 아이템(영화)과 상호작용이 있을 확률을 예측하게 하였다. 
Binary이므로 Loss는 CrossEntropy로 계산하였다.

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

### NDCG
- NDCG는 추천시스템에 주로 사용되는 평가지표이다.
- Cumulative Gain (CG)
$$CG_p=\sum_{i=1}^{p} rel_i\;\;\;\;p:particular \;rank\; position$$
CG는 특정 순위 p까지의 relevence의 합이다. 
relevence는 user와 item과의 상관관계를 나타내는 값이다. 일반적으로 user가 item에 매긴 점수 등이고 이 논문에서는 상호작용이 있으면 1 없으면 0이다. 즉
$$rel_i\in\{0,1\}$$

- Discounted Cumulative Gain (DCG)
$$DCG_p=\sum_{i=1}^{p}\frac{rel_i}{\log_2(i+1)}$$
DCG는 순위가 낮을 수록 relevence값을 차감시켜  높은 순위를 잘 추정하는 것이 중요한 평가지표이다.
아래 식은 높은 관련성 점수를 가진 아이템을 낮은 순위로 평가했을 때 더 큰 페널티를 주는 평가지표이다. 하지만 이 논문에서는 relevence가 binary이므로 두 값은 같다.
$$DCG_p=\sum_{i=1}^{p}\frac{2^{rel_i}-1}{\log_2(i+1)}$$

- Normalized DCG (NDCG)
$$NDCG_p=\frac{DCG_p}{IDCG_p}$$
NDCG는 DCG를 IDCG로 나눈 값이다. IDCG는 relevence가 높은 순으로 순위를 매긴 이상적인 상황일 때의 DCG값이다. NDCG는 0~1사이 값을 가진다.
이 논문에서 IDCG는 상호작용이 있는 아이템을 1순위로 한 경우이므로 
$$IDCG_p=\frac{1}{\log_2(1+1)}=1\Rightarrow NDCG_p=DCG_p$$
DCG는 상호작용이 있는 아이템을 i순위로 하였을 때 다음과 같이 계산된다.
$$DCG_p=\frac{1}{\log_2(i+1)}=\frac{\log2}{\log(i+1)}$$














