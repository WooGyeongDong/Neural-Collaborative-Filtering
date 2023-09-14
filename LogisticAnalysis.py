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
beta_hat = np.random.normal(0,1, size=(len(beta)))
lr = 1e-5

def gradient(x, y, b):
    return x.T@(-y+1/(np.exp(-x@b)+1))


def loss(x, y, b):
    delta = 1e-6
    cost = (-y*np.log(1/(1+np.exp(-x@b))+delta)-(1-y)*np.log(1-1/(1+np.exp(-x@b))+delta)).mean()
    return cost 

error = []
iteration = 100000
for i in range(iteration):
    beta_hat = beta_hat - lr * gradient(X, y, beta_hat)
    error.append(loss(X,y,beta_hat))
    
    if i % 1000 == 0:
        print(beta_hat, loss(X,y,beta_hat))
    
print("Learned beta: {} \nTrue beta: {})".format(np.round(beta_hat,4), beta))


import matplotlib.pyplot as plt
plt.plot(error)
plt.plot(error[7000:])



# from sklearn.linear_model import LogisticRegression  
  
# model = LogisticRegression(fit_intercept=True, penalty=None)  
# model.fit(X, y)
# print(model.coef_)
# print("{:5f},{:5f},{:5f}".format(model.coef_[0,0],model.coef_[0,1],model.coef_[0,2]))
