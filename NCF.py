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
####load the MovieLens 1m dataset in a Pandas dataframe
ratings = pd.read_csv('ml-1m/ratings.dat', delimiter='::', header=None, 
    names=['user_id', 'movie_id', 'rating', 'timestamp'], 
    usecols=['user_id', 'movie_id'], engine='python')  ##encoding latin1

users = pd.read_csv('ml-1m/users.dat', delimiter='::', header=None, 
    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'], 
    usecols=['user_id', 'gender', 'age', 'occupation'], engine='python')

movies = pd.read_csv('ml-1m/movies.dat', delimiter='::', header=None, 
    names=['movie_id', 'title', 'genres'], 
    usecols=['movie_id', 'genres'], engine='python')
#%%
####영화를 장르로 one-hot encoding
movies_genre_iter = (set(x.split(',')) for x in movies.genres)
movies_genre_set = sorted(set.union(*movies_genre_iter))
indicator_mat = pd.DataFrame(np.zeros((len(movies), len(movies_genre_set))), columns=movies_genre_set)
for i, genre in enumerate(movies.genres):
    indicator_mat.loc[i, genre.split(',')] = 1
    
movies_onehot = movies.join(indicator_mat).drop(columns=['genres'])

#### user one-hot encoding
users_onehot = pd.get_dummies(users, columns=['gender', 'age', 'occupation'])
#%%
##### user*movie interaction matrix
# y_mat = pd.DataFrame(np.zeros((len(users), len(movies))), columns=movies.movie_id, index=users.user_id)
# for i in range(len(ratings)):
#     y_mat.loc[ratings.user_id.iloc[i], ratings.movie_id.iloc[i]] = 1   
    
# y_mat.to_csv("ymat.csv")            
y_mat = pd.read_csv('ymat.csv', index_col=0)
#%%
### y와 user, movie 매칭
user = pd.DataFrame(np.repeat(users_onehot.values, len(movies_onehot), axis=0))
user.columns = users_onehot.columns
user = user.drop(columns=['user_id'])

item = pd.concat([movies_onehot]*len(users_onehot)).drop(columns=['movie_id'])

target = pd.DataFrame(y_mat.to_numpy().flatten())
target.columns = ["y"]

# all = pd.DataFrame()
# for i in range(100):
#     if target.iloc[i,0] == 1 :
#         temp = pd.concat([user.drop(columns=['user_id']).iloc[i],item.drop(columns=['movie_id']).iloc[i,:],target.iloc[i,:]])
#         all = pd.concat([all,temp], axis=1)

# all = pd.concat([user.drop(columns=['user_id']),item.drop(columns=['movie_id']),target],axis=1)

#%%
###########헛짓거리
class CustomDataset(Dataset):
    def __init__(self, user, item, target):
        self.user = user.values
        self.item = item.values
        self.target = target.values.reshape(-1,1)
        
    def __len__(self):
        return len(self.user)
    
    def __getitem__(self, index):
        user = torch.FloatTensor(self.user[index])
        item = torch.FloatTensor(self.item[index])
        target = torch.FloatTensor(self.target[index])
        return user, item, target

# user_train = user.drop(columns=['user_id'])
# item_train = item.drop(columns=['movie_id'])
# target_train = target
                       
dataset = CustomDataset(user, item, target)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

data = torch.zeros((1,49))
for batch_idx, samples in enumerate(dataloader):
    user, item, target = samples
    all = torch.cat([user, item, target], dim = 1)
    positive = all[target.squeeze() == 1]
    negative = all[target.squeeze() == 0]
    negative = negative[:len(positive)*4]
    data = torch.cat([data, positive, negative], dim = 0)
    
    if batch_idx == 100 :
        break
data = data[1:]
data.shape
#%%
#######################

class NCFDataset(Dataset):
    def __init__(self, data):
        self.user = data[:,:30]
        self.item = data[:,30:-1]
        self.target = data[:,-1]
        
    def __len__(self):
        return len(self.user)
    
    def __getitem__(self, index):
        user = torch.FloatTensor(self.user[index])
        item = torch.FloatTensor(self.item[index])
        target = torch.FloatTensor(self.target[index])
        return user, item, target


train_dataset = NCFDataset(data)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)


#%%
class MLP(torch.nn.Module):
    def __init__(self, user_d, item_d, predictive_factor):
        super().__init__()
        self.layer = nn.ModuleList()
        self.user_embedding = nn.Linear(user_d, 16)
        self.item_embedding = nn.Linear(item_d, 16)
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

        return out
#%%
  
class GMF(torch.nn.Module):
    def __init__(self, user_d, item_d) -> None:
        super().__init__()
        self.user_embedding = nn.Linear(user_d, 16)
        self.item_embedding = nn.Linear(item_d, 16)
        self.outlayer = nn.Linear(16,1)
        
        
    def forward(self, U, I):
        p = self.user_embedding(U)
        q = self.item_embedding(I)
        out = p*q
        out = self.outlayer(out) 
        out = F.sigmoid(out)
        
        return out
#%%

model = MLP(30, 18, 8)
model = GMF(30, 18) 
            
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%

total_loss = []
nb_epochs = 100               
for epoch in range(nb_epochs + 1):
    sum_loss = 0
    for samples in train_dataloader:
      
        user_train, item_train, y_train = samples
          
          # prediction 계산
        prediction = model(user_train, item_train)

        # loss 계산
        loss = nn.BCELoss()(prediction.squeeze(), y_train)

        # parameter 조정
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        
        
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(
        epoch, nb_epochs, sum_loss/len(train_dataloader)))
    total_loss.append(sum_loss/len(train_dataloader))
      
#%%       
import matplotlib.pyplot as plt
plt.plot(total_loss)
# %%
class NCF(torch.nn.Module):
    def __init__(self, user_d, item_d, predictive_factor):
        super().__init__()
        self.layer = nn.ModuleList()
        self.MLP_user_embedding = nn.Linear(user_d, 16)
        self.MLP_item_embedding = nn.Linear(item_d, 16)
        self.GMF_user_embedding = nn.Linear(user_d, 16)
        self.GMF_item_embedding = nn.Linear(item_d, 16)
        self.layer.append(nn.Linear(32, predictive_factor*4))
        self.layer.append(nn.Linear(predictive_factor*4, predictive_factor*2))
        self.layer.append(nn.Linear(predictive_factor*2, predictive_factor))
        self.layer.append(nn.Linear(predictive_factor+16, 1))

    def forward(self,U,I):
        p_g = self.GMF_user_embedding(U)
        q_g = self.GMF_item_embedding(I)
        p_m = self.MLP_user_embedding(U)
        q_m = self.MLP_item_embedding(I)
        out_m = torch.cat((p_m,q_m),1)
        
        for layer in self.layer[:-1] :
            out_m = F.relu(layer(out_m)) 
        
        out_g = p_g*q_g
        
        out = torch.cat((out_g, out_m),1)
            
        out = F.sigmoid(self.layer[-1](out))

        return out
#%%

model = NCF(30, 18, 8)  
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
