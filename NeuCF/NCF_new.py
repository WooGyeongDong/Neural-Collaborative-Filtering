#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
#%%
####load the MovieLens 1m dataset in a Pandas dataframe
ratings = pd.read_csv('ml-1m/ratings.dat', delimiter='::', header=None, 
    names=['user_id', 'movie_id', 'rating', 'timestamp'], 
    usecols=['user_id', 'movie_id'], engine='python')  ##encoding latin1

#%%

test_rating = ratings.copy()
test_rating = test_rating.drop_duplicates(subset=['user_id'], keep='last')
test_rating
train_rating = ratings.copy()
train_rating = pd.concat([train_rating, test_rating])
train_rating = train_rating.drop_duplicates(keep=False)
val_rating = train_rating.copy()
val_rating = val_rating.sample(frac=1, ignore_index= True)
val_rating = val_rating.drop_duplicates(subset=['user_id'])
train_rating = pd.concat([train_rating, val_rating])
train_rating = train_rating.drop_duplicates(keep=False)

#%%
#### dummy X
class CustomDataset(Dataset):
    def __init__(self, total_ratings, ratings, num_negative):
        self.total_ratings = total_ratings
        self.ratings = ratings
        self.num_negative = num_negative
        self.users, self.items, self.labels = self.data()
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]
        
    def data(self):
        users, items, labels = [], [], []
        user_item_set = set(zip(self.ratings['user_id'], self.ratings['movie_id']))
        total_user_item_set = set(zip(self.total_ratings['user_id'], self.total_ratings['movie_id']))
        movie_ids = self.ratings.movie_id.unique()
        for u, i in tqdm(user_item_set):
            users.append(u)
            items.append(i)
            labels.append(1)
            tmp_check = []
            
            negative_ratio = self.num_negative
            for _ in range(negative_ratio):
                # random sampling
                negative_item = np.random.choice(movie_ids)
                # checking interaction
                while (u, negative_item) in total_user_item_set or negative_item in tmp_check:
                    negative_item = np.random.choice(movie_ids)
                users.append(u)
                items.append(negative_item)
                labels.append(0)
                tmp_check.append(negative_item)
        
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

#%%

n = 80
train_dataset = CustomDataset(ratings, train_rating, 4)
val_dataset = CustomDataset(ratings, val_rating, n)
test_dataset = CustomDataset(ratings, test_rating, n)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=n+1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=n+1, shuffle=False)


#%%

class MLP(torch.nn.Module):
    def __init__(self, user_d, item_d, predictive_factor):
        super().__init__()
        self.layer = nn.ModuleList()
        self.user_embedding = nn.Embedding(user_d, 16)
        self.item_embedding = nn.Embedding(item_d, 16)
        self.layer.append(nn.Linear(32, predictive_factor*4))
        self.layer.append(nn.Linear(predictive_factor*4, predictive_factor*2))
        self.layer.append(nn.Linear(predictive_factor*2, predictive_factor))
        self.layer.append(nn.Linear(predictive_factor, 1))

    def forward(self,U,I):
        p = self.user_embedding(U)
        q = self.item_embedding(I)
        out = torch.cat((p,q),1)
        
        for layer in self.layer[:-1] :
            out = F.relu(layer(out)) 
            
        out = F.sigmoid(self.layer[-1](out))

        return out.reshape(-1)
#%%

 
class GMF(torch.nn.Module):
    def __init__(self, user_d, item_d) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(user_d, 16)
        self.item_embedding = nn.Embedding(item_d, 16)
        self.outlayer = nn.Linear(16,1)
        
        
    def forward(self, U, I):
        p = self.user_embedding(U)
        q = self.item_embedding(I)
        out = p*q
        out = self.outlayer(out) 
        out = F.sigmoid(out)
        
        return out.reshape(-1)   

#%%

class NCF(torch.nn.Module):
    def __init__(self, user_d, item_d, predictive_factor):
        super().__init__()
        self.layer = nn.ModuleList()
        self.MLP_user_embedding = nn.Embedding(user_d, 16, bias=False)
        self.MLP_item_embedding = nn.Embedding(item_d, 16, bias=False)
        self.GMF_user_embedding = nn.Embedding(user_d, 16, bias=False)
        self.GMF_item_embedding = nn.Embedding(item_d, 16, bias=False)
        self.layer.append(nn.Linear(32, predictive_factor*4))
        self.layer.append(nn.Linear(predictive_factor*4, predictive_factor*2))
        self.layer.append(nn.Linear(predictive_factor*2, predictive_factor))
        self.MLP_outlayer = nn.Linear(predictive_factor, 1, bias=False)
        self.GMF_outlayer = nn.Linear(16, 1, bias=False)
        nn.init.normal_(self.layer[0].weight, mean=0, std=0.01)
        nn.init.normal_(self.layer[1].weight, mean=0, std=0.01)
        nn.init.normal_(self.layer[2].weight, mean=0, std=0.01)
        nn.init.normal_(self.MLP_user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.MLP_item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.GMF_user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.GMF_item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.MLP_outlayer.weight, mean=0, std=0.01)
        nn.init.normal_(self.GMF_outlayer.weight, mean=0, std=0.01)

    def forward(self, U, I, pretrain):
        if pretrain == "mlp" :
            p_m = self.MLP_user_embedding(U)
            q_m = self.MLP_item_embedding(I)
            out_m = torch.cat((p_m,q_m),1)
            for layer in self.layer :
                out_m = F.relu(layer(out_m))
                
            out = F.sigmoid(self.MLP_outlayer(out_m))
        
        if pretrain == "gmf" :    
            p_g = self.GMF_user_embedding(U)
            q_g = self.GMF_item_embedding(I)
            out_g = p_g*q_g
            out = F.sigmoid(self.GMF_outlayer(out_g))
            
        if pretrain == "ncf" :
            p_m = self.MLP_user_embedding(U)
            q_m = self.MLP_item_embedding(I)
            out_m = torch.cat((p_m,q_m),1)
            for layer in self.layer :
                out_m = F.relu(layer(out_m))
            out_m = self.MLP_outlayer(out_m)
            
            p_g = self.GMF_user_embedding(U)
            q_g = self.GMF_item_embedding(I)
            out_g = p_g*q_g
            
            out_g = self.GMF_outlayer(out_g)
                
            out = F.sigmoid((0.5*out_m + 0.5*out_g))
            
        return out.reshape(-1)
##### model evaluation
def evaluation(dataloader):
    hit = []
    ndcg = []
    for samples in dataloader:
        
        user_test, item_test, y_test = samples
        y_test = y_test.float()
        
        # prediction 계산
        prediction = model(user_test, item_test)
        prediction = prediction.tolist()
        ranking = sorted(prediction, reverse=True)
        if ranking[9] <= prediction[0]:
            hit.append(1)
            rank = ranking.index(prediction[0]) + 1
            ndcg.append(1/np.log2(rank+1))
            
    return sum(hit)/len(dataloader), sum(ndcg)/len(dataloader)

#%%

model = MLP(6041, 4000, 8)
model = GMF(6041, 4000)  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# %%
nb_epochs = 10
best_hr = 0
patience_limit = 3 # 몇 번의 epoch까지 지켜볼지를 결정
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
val = []
total_loss = []
            
for epoch in tqdm(range(nb_epochs)):
    sum_loss = 0
    for samples in train_dataloader:
      
        user_train, item_train, y_train = samples
          
        # prediction 계산
        prediction = model(user_train, item_train)

        # loss 계산
        y_train = y_train.float()
        loss = nn.BCELoss()(prediction, y_train)

        # parameter 조정
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        
    total_loss.append(sum_loss/len(train_dataloader))   
     
    if epoch % 10 == 0:

        ### Validation loss Check
        val_loss = 0
        for samples in val_dataloader:
            
            user_val, item_val, y_val = samples

            val_pred = model(user_val.reshape(-1,1), item_val.reshape(-1,1))
            loss = nn.BCELoss()(val_pred, y_val.reshape(-1,1))
            val_loss += loss.item()
        
        hr, ndcg10 = evaluation(val_dataloader)
        val.append(val_loss/len(train_dataloader))
        
        print('Epoch {:4d}/{} val_Loss: {:.6f} HR: {:6f} NDCG: {:6f}'.format(
        epoch, nb_epochs, sum_loss/len(train_dataloader), hr, ndcg10))
        
        ### early stopping 여부를 체크하는 부분 ###
        if abs(hr - best_hr) < 1e-5: # loss가 개선되지 않은 경우
        # if val_loss > best_loss :
            patience_check += 1

            if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                print("Learning End. Best_HR:{:6f}".format(best_hr))
                break

        else: # loss가 개선된 경우
            best_hr = hr
            best_model = copy.deepcopy(model)
            patience_check = 0
      
#%%       
import matplotlib.pyplot as plt
plt.plot(total_loss)
# %%