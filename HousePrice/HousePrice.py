#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import copy
#%%

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data 추출
train_target = train_data['SalePrice']

train_data = train_data.drop(columns=['Id','SalePrice'])

test_data = test_data.drop(columns=['Id'])

train_target = np.log(train_target)

# train data와 test data가 같은 더미변수를 가지기 위해 합침
all_data = pd.concat((train_data, test_data))

# 결측값은 0으로 채우고 더미변수로 변환
numerical_features = all_data.dtypes[all_data.dtypes!='object'].index
all_data[numerical_features] = all_data[numerical_features].fillna(0)
all_data = pd.get_dummies(all_data, dummy_na = True, drop_first = True)

# train data와 test data 분리
train_data = all_data.iloc[:1460,:]
test_data = all_data.iloc[1460:,:]


# vaildation dataset split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.25, random_state= 8)

#%%
# Dataset class 선언
class CustomDataset(Dataset):
	def __init__(self, data, target):

		self.inp = data.values
		self.outp = target.values.reshape(-1,1)

	def __len__(self):
		return len(self.inp)

	def __getitem__(self,idx):
		inp = torch.FloatTensor(self.inp[idx])
		outp = torch.FloatTensor(self.outp[idx])
		return inp, outp # 해당하는 idx(인덱스)의 input과 output 데이터를 반환한다.

# Train data와 Validation data를 DataLoader에 넣음
dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_valid, y_valid)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
# 학습 모델
class Net(torch.nn.Module):
    def __init__(self, layer_value):
        super(Net,self).__init__()
        self.layer = nn.ModuleList()
        for i, value in enumerate(layer_value[:-1]) :
            self.layer.append(nn.Linear(layer_value[i],layer_value[i+1]))
            nn.init.xavier_uniform_(self.layer[i].weight)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
      out = x
      for layer in self.layer[:-1] :
          out = F.gelu(layer(out))
          out = self.dropout(out)

      output = torch.relu(self.layer[-1](out))

      return output
  
# model과 optimizer 선언
layer_value = [288,1024,1024,1]
model = Net(layer_value).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# log에 0이 들어가지 않게 조정 후 loss 반환
def MSLE_loss(pred, target):
    log_pred = torch.log(pred + 0.1)
    loss = nn.MSELoss()(log_pred, target)
    return loss
#%%
nb_epochs = 500
best_loss = 10 ** 9 # 매우 큰 값으로 초기값 가정
patience_limit = 3 # 몇 번의 epoch까지 지켜볼지를 결정
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
val = []

for epoch in range(nb_epochs + 1):
  sum_loss = 0
  model.train()
  for batch_idx, samples in enumerate(dataloader):


    x_train, y_train = samples
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # prediction 계산
    prediction = model(x_train)

    # loss 계산
    loss = MSLE_loss(prediction, y_train)

    # parameter 조정
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()

  scheduler.step()

  # 10번 epoch마다 평균 loss 출력
  if epoch % 10 == 0:
    print('Epoch {:4d}/{} Loss: {:.6f}'.format(
        epoch, nb_epochs, sum_loss/len(dataloader)))

  ### Validation loss Check
  model.eval()
  val_loss = 0
  for x_val, y_val in val_dataloader:

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    val_pred = model(x_val)
    loss = MSLE_loss(val_pred, y_val)

    val_loss += loss.item()
  val.append(val_loss)
  ### early stopping 여부를 체크하는 부분 ###
  if abs(val_loss - best_loss) < 1e-3: # loss가 개선되지 않은 경우
  # if val_loss > best_loss :
      patience_check += 1

      if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
          print("Learning End. Best_Loss:{:6f}".format(best_loss/len(val_dataloader)))
          break

  else: # loss가 개선된 경우
      best_loss = val_loss
      best_model = copy.deepcopy(model)
      patience_check = 0
      
#%%      
import matplotlib.pyplot as plt
plt.plot(val[:])
plt.plot(range(10,len(val)),val[10:])
plt.hist(train_data['SalePrice'])

plt.hist(train_target)

# Model Test
test_data_t = torch.tensor(test_data.values, dtype=torch.float32).to(device)
prediction = best_model(test_data_t)

# print(test_data)
print(prediction)

save = prediction.cpu()
save = save.detach().numpy()
pd.DataFrame(save).to_csv("result.csv")

area = train_data['LotArea']
plt.hist(area, range=(0,50000))

street = train_data['RoofStyle']
street.value_counts().plot(kind='bar')

##########################################################
#%%
class Learning():
  def __init__(self, dataloader, model, loss, optimizer):
      self.dataloader = dataloader
      self.model = model
      self.loss = loss
      self.optimizer = optimizer
      
  def train(self, valid=False):
    sum_loss = 0
    if valid :
      model.eval()
    else:
      model.train()
    for samples in enumerate(self.dataloader):
      x_train, y_train = samples
      x_train = x_train.to(device)
      y_train = y_train.to(device)

      # prediction 계산
      prediction = self.model(x_train)

      # loss 계산
      loss = self.loss(prediction, y_train)
      sum_loss += loss.item()
      
      if valid : continue
      # parameter 조정
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    
    mean_loss = sum_loss/len(self.dataloader) 
    print("mean Loss {:6f}".format(mean_loss))
    return mean_loss
  
  
train = Learning(dataloader, model, MSLE_loss, optimizer)  
valid = Learning(val_dataloader, model, MSLE_loss, optimizer)    
scheduler = optim.lr_scheduler.StepLR(train.optimizer, step_size=10, gamma=0.5)      
#%%
nb_epochs = 5
for epoch in range(nb_epochs + 1):
  train.train()
  
  scheduler.step()
  
  valid.train(True)

# %%
