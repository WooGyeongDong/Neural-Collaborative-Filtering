#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import copy
#%%
####load the MovieLens 1m dataset in a Pandas dataframe
ratings = pd.read_csv('ml-1m/ratings.dat', delimiter='::', header=None, 
    names=['user_id', 'movie_id', 'rating', 'timestamp'], 
    usecols=['user_id', 'movie_id'], engine='python')  ##encoding latin1
movies = pd.read_csv('ml-1m/movies.dat', delimiter='::', header=None, 
    names=['movie_id', 'title', 'genre'], 
    usecols=['movie_id'], engine='python')
#%%
test_rating = ratings.copy()
test_rating = test_rating.drop_duplicates(subset=['user_id'], keep='last')

train_rating = ratings.copy()
train_rating = pd.concat([train_rating, test_rating])
train_rating = train_rating.drop_duplicates(keep=False)
val_rating = train_rating.copy()
val_rating = val_rating.sample(frac=1, ignore_index= True)
val_rating = val_rating.drop_duplicates(subset=['user_id'])
train_rating = pd.concat([train_rating, val_rating])
train_rating = train_rating.drop_duplicates(keep=False)
movie_ids = movies.movie_id.unique()

#%%
class CustomDataset(Dataset):
    def __init__(self, total_ratings, ratings, movie_ids, num_negative):
        self.total_ratings = total_ratings
        self.ratings = ratings
        self.num_negative = num_negative
        self.movie_ids = movie_ids
        self.users, self.items, self.labels = self.data()
        
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]
        
    def data(self):
        users, items, labels = [], [], []
        user_item_set = set(zip(self.ratings['user_id'], self.ratings['movie_id']))
        total_user_item_set = set(zip(self.total_ratings['user_id'], self.total_ratings['movie_id']))
        for u, i in tqdm(user_item_set):
            users.append(u)
            items.append(i)
            labels.append(1)
            tmp_check = []
            
            negative_ratio = self.num_negative
            for _ in range(negative_ratio):
                # random sampling
                negative_item = np.random.choice(self.movie_ids)
                
                # checking interaction
                while (u, negative_item) in total_user_item_set or negative_item in tmp_check:
                    negative_item = np.random.choice(self.movie_ids)
                    
                users.append(u)
                items.append(negative_item)
                labels.append(0)
                tmp_check.append(negative_item)
        
        return torch.tensor(users), torch.tensor(items), torch.FloatTensor(labels)

#%%
n = 99
train_dataset = CustomDataset(ratings, train_rating, movie_ids, 4)
val_dataset = CustomDataset(ratings, val_rating, movie_ids, n)
test_dataset = CustomDataset(ratings, test_rating, movie_ids, n)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=n+1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=n+1, shuffle=False)


#%%

class NCF(torch.nn.Module):
    def __init__(self, user_d, item_d, predictive_factor):
        super().__init__()
        self.layer = nn.ModuleList()
        self.MLP_user_embedding = nn.Embedding(user_d, 16)
        self.MLP_item_embedding = nn.Embedding(item_d, 16)
        self.GMF_user_embedding = nn.Embedding(user_d, 16)
        self.GMF_item_embedding = nn.Embedding(item_d, 16)
        self.layer.append(nn.Linear(32, predictive_factor*4))
        self.layer.append(nn.Linear(predictive_factor*4, predictive_factor*2))
        self.layer.append(nn.Linear(predictive_factor*2, predictive_factor))
        self.MLP_outlayer = nn.Linear(predictive_factor, 1)
        self.GMF_outlayer = nn.Linear(16, 1)
        nn.init.normal_(self.layer[0].weight, mean=0, std=0.01)
        nn.init.normal_(self.layer[1].weight, mean=0, std=0.01)
        nn.init.normal_(self.layer[2].weight, mean=0, std=0.01)
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
            out_g = self.GMF_outlayer(out_g)
            out = F.sigmoid(out_g)
            
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
#%%
class MLP(torch.nn.Module):
    def __init__(self, user_d, item_d, predictive_factor):
        super().__init__()
        self.layer = nn.ModuleList()
        self.user_embedding = nn.Embedding(user_d, 32)
        self.item_embedding = nn.Embedding(item_d, 32)
        self.layer.append(nn.Linear(64, predictive_factor*4))
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
        self.user_embedding = nn.Embedding(user_d, 32)
        self.item_embedding = nn.Embedding(item_d, 32)
        self.outlayer = nn.Linear(32,1)
        nn.init.normal_(self.outlayer.weight, mean=0, std=0.01)
        
    def forward(self, U, I):
        p = self.user_embedding(U)
        q = self.item_embedding(I)
        out = p*q
        out = self.outlayer(out) 
        out = F.sigmoid(out)
        
        return out.reshape(-1)
#%%   
class NCF(nn.Module):
    def __init__(self, user_d, item_d, predictive_factor) -> None:
        super().__init__()
        self.gmf = GMF(user_d, item_d)
        self.mlp = MLP(user_d, item_d, predictive_factor)
        self.output = nn.Linear(predictive_factor+32, 1)
        
    def forward(self, U, I):
        out_g = self.gmf(U, I)
        out_m = self.mlp(U, I)
        out  = torch.cat((out_g, out_m),1)
        out = self.output(out)
        out = F.sigmoid(out)
        
        return out.reshape(-1)


#%% 
class NCF(torch.nn.Module):
    def __init__(self, user_d, item_d, predictive_factor):
        super().__init__()
        self.layer = nn.ModuleList()
        self.MLP_user_embedding = nn.Embedding(user_d, 32)
        self.MLP_item_embedding = nn.Embedding(item_d, 32)
        self.GMF_user_embedding = nn.Embedding(user_d, 32)
        self.GMF_item_embedding = nn.Embedding(item_d, 32)
        self.layer.append(nn.Linear(64, predictive_factor*4))
        self.layer.append(nn.Linear(predictive_factor*4, predictive_factor*2))
        self.layer.append(nn.Linear(predictive_factor*2, predictive_factor))
        self.layer.append(nn.Linear(predictive_factor+32, 1))

        
    def pretrain(self, mlp_path, gmf_path):
        mlp_model = MLP(6041, 3953, 8)
        mlp_model.load_state_dict(torch.load(mlp_path))
        gmf_model = GMF(6041, 3953)
        gmf_model.load_state_dict(torch.load(gmf_path))
        self.MLP_user_embedding.weight.data = mlp_model.user_embedding.weight.data
        self.MLP_item_embedding.weight.data = mlp_model.item_embedding.weight.data
        self.GMF_user_embedding.weight.data = gmf_model.user_embedding.weight.data
        self.GMF_item_embedding.weight.data = gmf_model.item_embedding.weight.data
        for i, layer in enumerate(self.layer):
            layer.weight.data = mlp_model.layer[i].weight.data
            layer.bias.data = mlp_model.layer[i].bias.data
            
        self.layer[-1].weight.data = 0.5 * torch.cat([mlp_model.layer[-1].weight.data, gmf_model.outlayer.weight.data], dim=-1)
        self.layer[-1].bias.data = 0.5 * (mlp_model.layer[-1].bias.data + gmf_model.outlayer.bias.data)

        
    def forward(self,U,I):
        p_g = self.GMF_user_embedding(U)
        q_g = self.GMF_item_embedding(I)
        p_m = self.MLP_user_embedding(U)
        q_m = self.MLP_item_embedding(I)
        out_m = torch.cat((p_m,q_m),1)
        
        for layer in self.layer[:-1] :
            out_m = F.relu(layer(out_m)) 
        
        out_g = torch.mul(p_g,q_g)
        
        out = torch.cat((out_g, out_m),1)
            
        out = F.sigmoid(self.layer[-1](out))

        return out.reshape(-1)
#%%
##### model evaluation
def evaluation(eval_model, dataloader):
    hit = []
    ndcg = []
    # sum_loss = 0
    for samples in dataloader:
        
        user_test, item_test, y_test = samples
        y_test = y_test.float()
        
        # prediction 계산
        prediction = eval_model(user_test, item_test)
        # loss = nn.BCELoss()(prediction, y_test)
        # sum_loss += loss.item()
        prediction = prediction.tolist()
        ranking = sorted(prediction, reverse=True)
        if ranking[9] <= prediction[0]:
            hit.append(1)
            rank = ranking.index(prediction[0]) + 1
            ndcg.append(1/np.log2(rank+1))
            
    return sum(hit)/len(dataloader), sum(ndcg)/len(dataloader)#, sum_loss/len(dataloader)
    

# %%
def train(model, optimizer, dataloader, valid_dataloader, nb_epochs, model_path) :
    best_hr = 0
    total_loss = []
    hr = []
    ndcg10 = []
    
    hrtemp, ndcg10temp = evaluation(model, valid_dataloader)
    hr.append(hrtemp)
    ndcg10.append(ndcg10temp)    
    print('Epoch {:4d}/{}  HR: {:6f} NDCG: {:6f}'.format(
    0, nb_epochs, hrtemp, ndcg10temp))   
             
    for epoch in tqdm(range(nb_epochs)):
        sum_loss = 0
        
        
        for samples in dataloader:
        
            user_train, item_train, y_train = samples
            
            # prediction 계산
            prediction = model(user_train, item_train)

            # loss 계산
            loss = nn.BCELoss()(prediction, y_train)

            # parameter 조정
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            
        total_loss.append(sum_loss/len(dataloader))   
        
        ### Validation
        
        hrtemp, ndcg10temp = evaluation(model, valid_dataloader)
        hr.append(hrtemp)
        ndcg10.append(ndcg10temp)
        
        print('Epoch {:4d}/{}  HR: {:6f} NDCG: {:6f}'.format(
        epoch+1, nb_epochs, hrtemp, ndcg10temp))
        
        if hrtemp > best_hr: 
            best_hr = hrtemp
            best_model = copy.deepcopy(model)
    torch.save(best_model.state_dict(), "{}.pth".format(model_path))        
    return total_loss, hr, ndcg10
        


      
#%%       



model_mlp = MLP(6041, 3953, 8)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)
mlp_loss, mlp_hr, mlp_ndcg = train(model_mlp, optimizer, train_dataloader, val_dataloader, 50,"mlp_model")
pd.DataFrame([mlp_loss, mlp_hr, mlp_ndcg]).to_csv('mlp_result.csv')


model_gmf = GMF(6041, 3953)   
optimizer = torch.optim.Adam(model_gmf.parameters(), lr=1e-3)
gmf_loss, gmf_hr, gmf_ndcg = train(model_gmf, optimizer, train_dataloader, val_dataloader, 50, "gmf_model")
pd.DataFrame([gmf_loss, gmf_hr, gmf_ndcg]).to_csv('gmf_result.csv')

model_ncf = NCF(6041, 3953, 8)
optimizer = torch.optim.Adam(model_ncf.parameters(), lr=1e-3)
ncf_loss, ncf_hr, ncf_ndcg = train(model_ncf, optimizer, train_dataloader, val_dataloader, 50, "ncf_model")
pd.DataFrame([ncf_loss, ncf_hr, ncf_ndcg]).to_csv('ncf_result.csv')

model = NCF(6041, 3953, 8)
model.pretrain('mlp_model.pth', 'gmf_model.pth')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss, hr, ndcg = train(model, optimizer, train_dataloader, val_dataloader, 50, "pretrain_ncf_model")
pd.DataFrame([loss, hr, ndcg]).to_csv('pretrain_ncf_result.csv')


# %%

plt.plot(mlp_loss, label="MLP")
plt.plot(gmf_loss, label="GMF")
plt.plot(ncf_loss, label="NCF")
plt.plot(loss, label="Pretrained NCF")
plt.xlabel("iter")
plt.ylabel("Train Loss")
plt.legend(loc = "upper right")
plt.show()

plt.plot(mlp_hr, label="MLP")
plt.plot(gmf_hr, label="GMF")
plt.plot(ncf_hr, label="NCF")
plt.plot(hr, label="Pretrained NCF")
plt.xlabel("iter")
plt.ylabel("HR@10")
plt.legend(loc = "lower right")
plt.show()

plt.plot(mlp_ndcg, label="MLP")
plt.plot(gmf_ndcg, label="GMF")
plt.plot(ncf_ndcg, label="NCF")
plt.plot(ndcg, label="Pretrained NCF")
plt.xlabel("iter")
plt.ylabel("NDCG@10")
plt.legend(loc = "lower right")
plt.show()
#%%
mlp_path = 'mlp_model.pth'
model_mlp.load_state_dict(torch.load(mlp_path))
model_mlp.user_embedding.weight.data
model.MLP_user_embedding.weight.data
h, n = evaluation(model_mlp, val_dataloader)
mr = pd.read_csv("mlp_result.csv")
ml = mr.loc[0].to_list()
ml.pop(0)
ml
mlp_hr.insert(0,h)
mlp_ndcg.insert(0,n)




####################################################################33

class NeuMF(torch.nn.Module):
    def __init__(self, num_users, num_items, latent_dim_mf, latent_dim_mlp, layers):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_mf = latent_dim_mf
        self.latent_dim_mlp = latent_dim_mlp

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1] + latent_dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.view(-1)


model = NeuMF(6041, 3953, 8, 8, [16,32,16,8])


class NCF(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, num_layers):
        super(NCF, self).__init__()
        

        # 임베딩 저장공간 확보; (num_embeddings, embedding_dim)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1))
        )

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        predict_size = factor_num
        self.predict_layer = nn.Linear(predict_size, 1)

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        # 임베딩 벡터 합치기
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        concat = output_MLP

        # 예측하기
        prediction = self.predict_layer(concat)
        prediction = F.sigmoid(prediction)
        return prediction.view(-1)

model = NCF(6041, 3953, 8, 3)

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, _ in test_loader:

        predictions = model(user, item)
        # 가장 높은 top_k개 선택
        _, indices = torch.topk(predictions, top_k)
        # 해당 상품 index 선택
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        # 정답값 선택
        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

metrics(model, val_dataloader, 10)
######################################################################

#%%
users = ratings.drop_duplicates(subset=['user_id']).drop(columns=['movie_id'])
users.index = users.user_id
movies = ratings.drop_duplicates(subset=['movie_id']).drop(columns=['user_id'])
movies.index = movies.movie_id
total = set(zip(ratings.user_id, ratings.movie_id))

user, item, y = pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 


for i in tqdm(range(len(ratings))):
    u = ratings.user_id.iloc[i]
    m = ratings.movie_id.iloc[i]

    user = pd.concat([user ,users.loc[u,:]], axis=1)
    item = pd.concat([item, movies.loc[m,:]], axis=1)
    y = pd.concat([y, pd.DataFrame([1])])

    #negative sampling    
    for _ in range(4):
        rv = random.choice(movies.values.squeeze())
        while((u, rv) in total) :
            rv = random.choice(movies.values.squeeze())
        user = pd.concat([user, users.loc[u,:]], axis=1)
        item = pd.concat([item, movies.loc[rv,:]], axis=1)
        y = pd.concat([y, pd.DataFrame([0])]) 
       
user = user.T
item = item.T

users, items, labels = [], [], []
user_item_set = set(zip(test_rating['user_id'], test_rating['movie_id']))
total_user_item_set = set(zip(ratings['user_id'], ratings['movie_id']))
movie_ids = ratings.movie_id.unique()
for u, i in tqdm(user_item_set):
    users.append(u)
    items.append(i)
    labels.append(1)
    tmp_check = []

    for _ in range(99):
        # random sampling
        negative_item = np.random.choice(movie_ids)
        # checking interaction
        while (u, negative_item) in total_user_item_set or negative_item in tmp_check:
            negative_item = np.random.choice(movie_ids)
        users.append(u)
        items.append(negative_item)
        labels.append(0)
        tmp_check.append(negative_item)


users, items, labels = torch.FloatTensor(users), torch.FloatTensor(items), torch.FloatTensor(labels)
len(users)
len(user_item_set)

hit = []
ndcg = []
for samples in test_dataloader:
    
    user_test, item_test, y_test = samples
    user_test = user_test.reshape(-1,1)
    item_test = item_test.reshape(-1,1)
    
    # prediction 계산
    prediction = model(user_test, item_test)
    prediction = prediction.tolist()
    ranking = sorted(prediction, reverse=True)
    if ranking[9] <= prediction[0]:
        hit.append(1)
        rank = ranking.index(prediction[0]) + 1
        ndcg.append(1/np.log2(rank+1))
        
sum(hit)/len(test_dataloader), sum(ndcg)/len(test_dataloader)

len(test_dataloader)

nn.Embedding(3761, 16)(item_test)