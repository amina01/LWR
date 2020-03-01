# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:35:53 2019

@author: Admin
"""



import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch


from sklearn.metrics import roc_auc_score as auc_roc
from sklearn.metrics import mean_squared_error as mse


from example import Example
from random import shuffle

import matplotlib.pyplot as plt #importing plotting module
from plotit import plotit


def eps_ins(y_true, y_pred):
    zero = torch.Tensor([0]) 
    eps=0.1
    return torch.max(zero, torch.abs(y_true-y_pred)-eps)



    
def hinge(y_true, y_pred):
    zero = torch.Tensor([0]) 
    return torch.max(zero, 1 - y_true * y_pred)

def create_ex_gauss(m1=1.0, m2=-1.0, sd=1.4, n=500):        #create example gaussian
    Xp =sd* np.random.randn(n,2)+m1
    Xn =sd* np.random.randn(n,2)+m2
    X = np.vstack((Xp,Xn))
    m = np.mean(X,axis=0)
    s = np.std(X,axis=0)
    Xp = (Xp-m)/s
    Xn = (Xn-m)/s
    data=[]
    Xp2=Xp
    Xn2=Xn
    for i in range(len(Xp)):
        ex=Example()
        ex.features_u=Xp2[i]
        ex.raw_features=Xp[i]
        ex.features_w=Xp2[i]
        ex.label=1.0
        ex.gamma=1.0#/len(Xp)
        data.append(ex)
    for i in range(len(Xn)):
        ex=Example()
        ex.features_u=Xn2[i]
        ex.raw_features=Xn[i]
        ex.features_w=Xn2[i]
        ex.label=-1.0
        ex.gamma=1.0#/len(Xn)
        data.append(ex)
    shuffle(data)
    return data
    



class Net(nn.Module):
    def __init__(self,d):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,1)        

    def forward(self,x):
        x = self.hidden1(x)

        return x
class RejNet(nn.Module):
    def __init__(self,d):
        super(RejNet, self).__init__()
        self.hidden1 = nn.Linear(d,2)   
        self.out = nn.Linear(d,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = F.tanh(x)        
        x = self.out(x)
        return x

############## Create Toy examples######################
examples=create_ex_gauss(m1=1.0, m2=2.0, sd=0.80, n=200)        
Xtr = np.array([e.features_w for e in examples])
Xtr=Variable(torch.from_numpy(Xtr)).type(torch.FloatTensor)
Ytr = np.array([e.label for e in examples])
Ytr=Variable(torch.from_numpy(Ytr)).type(torch.FloatTensor)
Ytr=Ytr[:,None]


############## Train classifier#########################
criterion1=hinge
class_epochs=25000
mlp_class=Net(Xtr.shape[1])
optimizer = optim.Adam(mlp_class.parameters())#, lr=0.01)

zero=np.zeros(Ytr.shape)
zero=Variable(torch.from_numpy(zero)).type(torch.FloatTensor)
mlp_rej=RejNet(Xtr.shape[1])
optimizer_rej = optim.Adam(mlp_rej.parameters(), lr=0.001)#, lr=0.01)#, lr=0.0001)
c=0.75
L = []
e=[]
print (mlp_rej.state_dict())

for epoch in range(class_epochs):
            # Forward pass: Compute predicted y by passing x to the model
    y_pred = mlp_class(Xtr)
   
    h = mlp_class(Xtr)
    r=mlp_rej(Xtr)
    e = eps_ins(Ytr,h)
    l1 = torch.max(r+e,c*(1-r))

    loss_r=torch.mean(torch.max(zero, l1 ))    
#    1/0

#   
    L.append(loss_r.data.numpy())

    
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    optimizer_rej.zero_grad()
    loss_r.backward()
    optimizer.step()
    optimizer_rej.step()

print (mlp_rej.state_dict())    
#import matplotlib.pyplot as plt
plt.close('all')
plt.plot(L)
plt.title('Loss')
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.figure()
plt.plot(r.detach().numpy()) 
plt.title('r') 
for param in mlp_class.parameters():
    param.requires_grad =False
        
for param in mlp_rej.parameters():
    param.requires_grad =False

y_p= mlp_class(Xtr)
y_p=y_p.detach()
y_r=mlp_rej(Xtr)
y_r=y_r.detach()
y_r=y_r.numpy().flatten()
y_p2=y_p.numpy().flatten()
Ytr=Ytr.numpy().flatten()
plt.figure()
plt.scatter(Ytr, y_p2)






test=create_ex_gauss(m1=1.0, m2=2.0, sd=1.0, n=200)     
X = np.array([e.features_w for e in test])
X=Variable(torch.from_numpy(X)).type(torch.FloatTensor)
Y = np.array([e.label for e in test])

y_p= mlp_class(X)
y_p=y_p.detach()
y_r=mlp_rej(X)
y_r=y_r.detach()
y_r=y_r.numpy().flatten()
y_p2=y_p.numpy().flatten()

auc_c=auc_roc(Y, y_p2)
auc_r=auc_roc(Y[y_r>0], y_p2[y_r>0])

ms=mse(Y, y_p2)
ms_r=mse(Y[y_r>0], y_p2[y_r>0])

print("MSE without rejection=", ms)
print("MSE with rejection=", ms_r)


print("AUC without rejection=", auc_c)
print("AUC with rejection=", auc_r)

print ("Number of examples rejected=", len(y_r[y_r<0]), "/", len(y_r))


#plt.close('all')
plt.figure()
X2 = np.array([e.raw_features for e in test])
plotit(X2,Y,clf=mlp_rej, transform = None, conts =[0], ccolors = ['g'], hold = False )
plt.title('test data')
plt.figure()
plotit(X2,Y,clf=mlp_class, transform = None, conts =[0], ccolors = ['k'])
plt.title('test data')

plt.figure()
Xtr=np.array(Xtr)
plt.scatter(Xtr[Ytr<0][:,0], Xtr[Ytr<0][:,1], marker='s')
plt.scatter(Xtr[Ytr>0][:,0], Xtr[Ytr>0][:,1], marker='o')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(['Positive class (+1)', 'Negative class (-1)'])
plt.title('train data')
plt.grid()

plt.figure()
plt.scatter(X2[Y<0][:,0], X2[Y<0][:,1], marker='s')
plt.scatter(X2[Y>0][:,0], X2[Y>0][:,1], marker='o')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(['Positive class (+1)', 'Negative class (-1)'])
plt.title('test data')
plt.grid()

#%%
Yts=Y
epsilon=0.1
import operator
ss=[[k,v,w ]for k, v, w in sorted(zip(y_r, Yts, y_p2), key=operator.itemgetter(0))]
ss2=np.array(ss)
frac=np.arange(0.0, 0.51, 0.05)
rmse=[]
DV2=[]
ind=(len(Yts)*np.arange(0.0, 0.51, 0.05)).astype(int)
for i in ind:
    ss1=np.array(ss[i:])
    rmse.append(auc_roc(ss1[:,1],ss1[:,2]))
    DV = []
    for _ in range(100):
        rand_y=np.ones(len(Yts))
        ll=np.random.choice(range(len(rand_y)),i , replace=False)
#        print(len(ll))
        rand_y[ll]=-1.0
#        if ll:
#        dv = np.mean(np.max((np.zeros(Yts[rand_y>0].shape),np.abs(Yts[rand_y>0]-y_p2[rand_y>0])-epsilon),axis=0))
        dv=auc_roc(Yts[rand_y>0], y_p2[rand_y>0])
        DV.append(dv)
    DV2.append(np.mean(DV))#print("loss unrej-rej random 20=",np.mean(DV),np.std(DV) )

#    1/0
plt.figure()
plt.plot(ind, rmse, marker='o', markersize=4)
plt.plot(ind, DV2, marker='s', markersize=4)
plt.xlabel('Number of rejections (Out of '+ str(len(Yts))+')')
plt.ylabel('AUC')
plt.legend(['Learned rejections', 'Random rejections'])
plt.grid()
