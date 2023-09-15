#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import random
#%%
####load the MovieLens 1m dataset in a Pandas dataframe
ratings = pd.read_csv('ml-1m/ratings.dat', delimiter='::', header=None, 
    names=['user_id', 'movie_id', 'rating', 'timestamp'], 
    usecols=['user_id', 'movie_id'], engine='python')  ##encoding latin1

#%%

test_rating = ratings.copy()
test_rating = test_rating.drop_duplicates(subset=['user_id'], keep='last')
train_rating = ratings.copy()
train_rating = pd.concat([train_rating, test_rating])
train_rating = train_rating.drop_duplicates(keep=False)

class CustomDataset(Dataset):
    def __init__(self, ratings, num_negative):
        self.ratings = ratings
        self.num_negative = num_negative
        self.users, self.items, self.labels = self.data()
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]
        
    def data(self):
        users = self.ratings.drop_duplicates(subset=['user_id']).drop(columns=['movie_id'])
        users_onehot = pd.get_dummies(users, columns=['user_id'])
        users_onehot.index = users.user_id
        movies = self.ratings.drop_duplicates(subset=['movie_id']).drop(columns=['user_id'])
        movies_onehot = pd.get_dummies(movies, columns=['movie_id'])
        movies_onehot.index = movies.movie_id
        total = set(zip(self.ratings.user_id, self.ratings.movie_id))
        
        user = pd.DataFrame()
        item = pd.DataFrame()  
        y = pd.DataFrame()
        for i in range(len(self.ratings)):
            u = self.ratings.user_id.iloc[i]
            m = self.ratings.movie_id.iloc[i]
        
            user = pd.concat([user ,users_onehot.loc[u,:]], axis=1)
            item = pd.concat([item, movies_onehot.loc[m,:]], axis=1)
            y = pd.concat([y, pd.DataFrame([1])])

            #negative sampling    
            for _ in range(self.num_negative):
                rv = random.choice(movies.values.squeeze())
                while((u, rv) in total) :
                    rv = random.choice(movies.values.squeeze())
                user = pd.concat([user, users_onehot.loc[u,:]], axis=1)
                item = pd.concat([item, movies_onehot.loc[rv,:]], axis=1)
                y = pd.concat([y, pd.DataFrame([0])]) 
                
        user = user.T
        item = item.T
        
        return torch.FloatTensor(user), torch.FloatTensor(item), torch.FloatTensor(y)

dataset = CustomDataset(ratings, 4)
dataset[1]

users = ratings.drop_duplicates(subset=['user_id'], keep='last').drop(columns=['movie_id'])
users_onehot = pd.get_dummies(users, columns=['user_id'])
users_onehot.index = users.user_id
movies = ratings.drop_duplicates(subset=['movie_id']).drop(columns=['user_id'])
movies_onehot = pd.get_dummies(movies, columns=['movie_id'])
movies_onehot.index = movies.movie_id


total = set(zip(ratings.user_id, ratings.movie_id))

#%%
user = pd.DataFrame()
item = pd.DataFrame()  
y = pd.DataFrame()
for i in range(10):
    u = ratings.user_id.iloc[i]
    m = ratings.movie_id.iloc[i]
   
    user = pd.concat([user ,users_onehot.loc[u,:]], axis=1)
    item = pd.concat([item, movies_onehot.loc[m,:]], axis=1)
    y = pd.concat([y, pd.DataFrame([1])])

    #negative sampling    
    for _ in range(4):
        rv = random.choice(movies.values.squeeze())
        while((u, rv) in total) :
            rv = random.choice(movies.values.squeeze())
        user = pd.concat([user, users_onehot.loc[u,:]], axis=1)
        item = pd.concat([item, movies_onehot.loc[rv,:]], axis=1)
        y = pd.concat([y, pd.DataFrame([0])]) 
            
#%%

last = 1
user = pd.DataFrame()
item = pd.DataFrame()
y = pd.DataFrame()

user_eval = pd.DataFrame()
item_eval = pd.DataFrame()
y_eval = pd.DataFrame()

check = []
i=1
j=0
for i in range(len(ratings)):

    user = pd.concat([user ,users_onehot.loc[ratings.iloc[i].user_id,:]], axis=1)
    item = pd.concat([item, movies_onehot.loc[ratings.movie_id.iloc[i],:]], axis=1)
    y = pd.concat([y, pd.DataFrame([1])])
    check.append(ratings.movie_id.iloc[i])
    
    for _ in range(len(check)*4):
        rv = random.choice(movies.index)
        if rv not in check:
            user = pd.concat([user, users_onehot.loc[last,:]])
            item = pd.concat([item, movies_onehot.loc[rv,:]])
            y = pd.concat([y, pd.DataFrame([0])])
            
           
                
    check = []
    last = ratings.user_id.iloc[i+1]
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

all = pd.DataFrame()
for i in range(len(target)):
    if target.iloc[i,0] == 1 :
        temp = pd.concat([user.iloc[i],item.iloc[i,:],target.iloc[i,:]])
        all = pd.concat([all,temp], axis=1)

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
            
        return out
#%%

model = NCF(30, 18, 8)  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# %%
nb_epochs = 100
best_loss = 10 ** 9 # 매우 큰 값으로 초기값 가정
patience_limit = 3 # 몇 번의 epoch까지 지켜볼지를 결정
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
val = []
total_loss = []
            
for epoch in range(nb_epochs + 1):
    sum_loss = 0
    for samples in train_dataloader:
      
        user_train, item_train, y_train = samples
          
        # prediction 계산
        prediction = model(user_train, item_train, "mlp")

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