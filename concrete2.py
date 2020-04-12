import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch



from sklearn.model_selection import train_test_split
from sklearn import metrics


import matplotlib.pyplot as plt #importing plotting module

def eps_ins(y_true, y_pred):
    zero = torch.Tensor([0]) 
    zero=Variable(zero).type(torch.cuda.FloatTensor)
    eps=10.0

    return torch.max(zero, torch.abs(y_true-y_pred)-eps)#
class Net(nn.Module):
    def __init__(self,d):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,64)
        self.hidden2 = nn.Linear(64,32)         
        self.out = nn.Linear(32,1)

    def forward(self,x):
#        x = x.view(x.size(0), -1)
        #x = F.tanh(x)
        x = self.hidden1(x)
#        x = F.tanh(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x=F.relu(x)
        x = self.out(x)
        return x
class RejNet(nn.Module):
    def __init__(self,d):
        super(RejNet, self).__init__()
        self.hidden1 = nn.Linear(d,64)   
        self.hidden2 = nn.Linear(64,32)   
        self.out = nn.Linear(32,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = F.tanh(x)
        x = self.hidden2(x)
        x = F.tanh(x)
        
        x = self.out(x)
        return x
        
    
mse_all=[]
mse_rej=[]
nrej=[]
runs=10
use_cuda=True
#from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
import pandas as pd
df=pd.read_excel('Concrete_Data.xls')
df_arr=np.array(df)
examples=df_arr[:,:-1]
labels=df_arr[:,-1]

import operator

for run in range(runs):    
    
        
    mse=[]
    corrs=[]
   
    
    Xtr, Xts, Ytr, Yts = train_test_split(examples, labels, test_size=0.1)
    
   
    
    Xtr=Variable(torch.from_numpy(Xtr)).type(torch.cuda.FloatTensor)
    
    Ytr=Variable(torch.from_numpy(Ytr)).type(torch.cuda.FloatTensor)
    
    
    Xts=Variable(torch.from_numpy(Xts)).type(torch.cuda.FloatTensor)
    
    Yts=Variable(torch.from_numpy(Yts)).type(torch.cuda.FloatTensor)


    class_epochs=15000
    mlp_class=Net(Xtr.shape[1])
    mlp_class.cuda()   
    optimizer = optim.Adam(mlp_class.parameters(), lr=0.005)
    
    mlp_rej=RejNet(Xtr.shape[1])
    mlp_rej.cuda()
    optimizer_rej = optim.Adam(mlp_rej.parameters(), lr=0.0005)
    scheduler_class = StepLR(optimizer, step_size=1000, gamma=0.75)
    scheduler_rej = StepLR(optimizer_rej, step_size=1000, gamma=0.9)
    c=1.0
    epsilon=1.0
    beta=1
    alpha=2
    L = []
    e=[]
#    print (mlp_rej.state_dict())
    bsize = len(Ytr)#200
    zero=np.zeros(bsize)
    zero=Variable(torch.from_numpy(zero)).type(torch.cuda.FloatTensor)
    for epoch in range(class_epochs):
        scheduler_class.step()
        scheduler_rej.step()
        s = np.random.choice(range(len(Ytr)),bsize, replace=False)
        h = mlp_class(Xtr[s]).squeeze()
        r=mlp_rej(Xtr[s]).squeeze()

        e=(Ytr[s]-h)**2
        
        l1 = torch.max(alpha/2*(r+e),c*(1-beta*r))
    
        loss_r=torch.mean(torch.max(zero, l1)) 

        L.append(loss_r)

        optimizer.zero_grad()
        optimizer_rej.zero_grad()
        loss_r.backward()
        optimizer.step()
        optimizer_rej.step()
        


    for param in mlp_class.parameters():
        param.requires_grad =False
            
    for param in mlp_rej.parameters():
        param.requires_grad =False

    y_p= mlp_class(Xts)
    y_p=y_p.detach()
    y_r=mlp_rej(Xts)
    y_r=y_r.detach()
    y_r=y_r.cpu().numpy().flatten()
    y_p2=y_p.cpu().numpy().flatten()
    
    
    Yts=Yts.cpu().numpy().flatten()
    
    
    ###########################################
    
    ss=[[k,v,w ]for k, v, w in sorted(zip(y_r, Yts, y_p2), key=operator.itemgetter(0))]
    ss2=np.array(ss)
    rmses_rej=[]
    for rejs in (np.array([0.1, 0.2, 0.3, 0.4, 0.5])*103).astype(int):
        ss1=np.array(ss[rejs:])
        y_p3=ss1[:,2]

        m_r=metrics.mean_squared_error(ss1[:,1],y_p3)
        rmses_rej.append(m_r)

    m_c=metrics.mean_squared_error(Yts, y_p2)
    print("accuracy without rejection=",m_c)
    

    
    
    
    print ("\n\nNumber of examples rejected=", len(y_r[y_r<=0]), "/", len(y_r))

    m_c=metrics.mean_squared_error(Yts, y_p2)
    mse_all.append(m_c)
    print("mse without rejection=",m_c)
    mse_rej.append(rmses_rej)
    print("mse with rejection=",rmses_rej)
    

        
print("\n\nMean MSE without rejection:", np.mean(mse_all),"+/-", np.std(mse_all))
print("Mean MSE with rejection:", np.mean(mse_rej, axis=0),"+/-", np.std(mse_rej, axis=0))

