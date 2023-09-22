import numpy as np
###### Data Generation

#Set beta
beta = np.array([0.4, -0.2, 0.3, 0.5, 0.6, -0.7])

# Generate X
n = 500
np.random.seed(seed=5)
X = np.random.normal(0,3, size=(n,len(beta)-1))
const = np.ones(n)
X = np.hstack((const.reshape(n,1),X))

# Generate y
mu = X@beta
prob = 1/(np.exp(-mu)+1)
y = prob.round()
# y = np.random.binomial(1, prob.flatten())


#Gradient Decent Algorithm
class LogisticRegression:
    def __init__(self, learning_rate, num_iter) -> None:
        self.learnig_rate = learning_rate
        self.num_iter = num_iter

    def gradient(self, x, y, b):
        return x.T@(-y+1/(np.exp(-x@b)+1))

    def loss(self, x, y, b):
        delta = 1e-6
        cost = (-y*np.log(1/(1+np.exp(-x@b))+delta)-(1-y)*np.log(1-1/(1+np.exp(-x@b))+delta)).mean()
        return cost 
    
    def fit(self, X, y):
        
        self.beta_hat = np.random.normal(0,1, size=(len(beta)))
        
        self.total_loss = []
        for i in range(self.num_iter):
            self.beta_hat -= self.learnig_rate * self.gradient(X, y, self.beta_hat)
            self.total_loss.append(self.loss(X, y, self.beta_hat))
            
            if i % 1000 == 0:
                print(self.beta_hat, self.loss(X, y, self.beta_hat))
        
    def predict(self, x):
        predict_prob = 1/(1+np.exp(-x@self.beta_hat))
        return predict_prob.round()
        
model = LogisticRegression(0.0001, 10000)
model.fit(X,y)

model.beta_hat
acc = np.mean(model.predict(X) == y)
acc
print("Learned beta: {} \nTrue beta: {})".format(np.round(model.beta_hat,4), beta))


import matplotlib.pyplot as plt
plt.plot(model.total_loss[:2000])



# from sklearn.linear_model import LogisticRegression  
  
# model = LogisticRegression(fit_intercept=True, penalty=None)  
# model.fit(X, y)
# print(model.coef_)
# print("{:5f},{:5f},{:5f}".format(model.coef_[0,0],model.coef_[0,1],model.coef_[0,2]))
