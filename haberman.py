# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:29:38 2019

@author: Amina Asif
"""



import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import StratifiedKFold as kf#train_test_split
from sklearn import metrics



def accuracy(Y,y_p):
    f, t, a=metrics.roc_curve(Y, y_p)
    AN=np.array(sum(x<=0 for x in Y))
    AP=np.array(sum(x>0 for x in Y))
    TN=(1.0-f)*AN
    TP=t*AP
    Acc2=(TP+TN)/len(Y)
    acc=max(Acc2)
    return acc

def hinge(y_true, y_pred):
    zero = torch.Tensor([0]) 
    return torch.max(zero, 1 - y_true * y_pred)
def eps_ins(y_true, y_pred):
    zero = torch.Tensor([0]) 
    zero=Variable(zero).type(torch.cuda.FloatTensor)
    eps=5.0
#    import pdb; pdb.set_trace()
    return torch.max(zero, torch.abs(y_true-y_pred)-eps)#
class Net(nn.Module):
    def __init__(self,d):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,2*d)
        self.hidden2 = nn.Linear(d*d,d*d)  
        self.hidden3 = nn.Linear(64,1)
        self.out = nn.Linear(2*d,1)

    def forward(self,x):
#        x = x.view(x.size(0), -1)
        #x = F.tanh(x)
        x = self.hidden1(x)
#        x = F.tanh(x)
        x = F.relu(x)
#        x = self.hidden2(x)
#        x=F.tanh(x)
#        x = self.hidden3(x)
        
        x = self.out(x)
        return x
class RejNet(nn.Module):
    def __init__(self,d):
        super(RejNet, self).__init__()
        self.hidden1 = nn.Linear(d,32)   
        self.hidden2 = nn.Linear(32,64)   
        self.out = nn.Linear(64,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = F.tanh(x)
        x = self.hidden2(x)
        x = F.tanh(x)
        
        x = self.out(x)
        return x
        
def readData():
    X=[]
    Y=[]
    file = open("haberman.data","r")
    data=file.readlines()
    file.close() 
#    shuffle(data)
    for i in range(len(data)):
        temp=data[i].split(',')
        temp[3]=temp[3][0]
        X.append(temp[:3])
        Y.append(temp[3])
    Y=np.array(Y,dtype=np.int)
    Y[Y==1]=-1
    Y[Y==2]=1
    X=np.array(X, dtype=np.float64)
    X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
   
    return X,Y    
mse_all=[]
mse_rej=[]
acc_all=[]
acc_rej=[]
nrej=[]
runs=10
for run in range(runs):    
    
        
    mse=[]
    corrs=[]
    
    file = open("haberman.data","r")
    data=file.readlines()
    examples=[]
    labels=[]
    for d in data:
        examples.append([float(i) for i in d.split(',')[:-1]])
        labels.append(int(d.split(',')[-1]))#d.split(',')[-1]])

    examples=np.array(examples)
    examples=(examples-np.min(examples,axis=0))/(np.max(examples,axis=0)-np.min(examples,axis=0))
    examples=(examples-np.mean(examples,axis=0))/np.std(examples,axis=0)
    labels=np.array(labels)
    labels[labels==2]=-1

    skf = kf(n_splits=5)#, random_state=40)
    for train, test in skf.split(examples, labels):
        Xtr=examples[train]
        Xts=examples[test]
        
        Ytr=labels[train]
        Yts=labels[test]
        Xtr=Variable(torch.from_numpy(Xtr)).type(torch.FloatTensor)
    
        Ytr=Variable(torch.from_numpy(Ytr)).type(torch.FloatTensor)
        
        
        Xts=Variable(torch.from_numpy(Xts)).type(torch.FloatTensor)
        
        Yts=Variable(torch.from_numpy(Yts)).type(torch.FloatTensor)
    
    
        class_epochs=10000
        mlp_class=Net(Xtr.shape[1])
#        mlp_class.cuda()   
        optimizer = optim.Adam(mlp_class.parameters(), lr=0.005)
        
        mlp_rej=RejNet(Xtr.shape[1])
#        mlp_rej.cuda()
        optimizer_rej = optim.Adam(mlp_rej.parameters(), lr=0.0005)#, lr=0.01)#, lr=0.0001)
        # gamma = decaying factor
        scheduler_class = StepLR(optimizer, step_size=1000, gamma=0.75)
        scheduler_rej = StepLR(optimizer_rej, step_size=1000, gamma=0.75)
        c=01.50#2.250#.50#370#0.15
#        epsilon=1.0
        beta=1#/(1-2*c)
        alpha=2#1.0
        
        L = []
        e=[]
    #    print (mlp_rej.state_dict())
        bsize = len(Ytr)#200
        zero=np.zeros(bsize)
        zero=Variable(torch.from_numpy(zero)).type(torch.FloatTensor)
        for epoch in range(class_epochs):
#            print("LR_class:",scheduler_class.get_lr())
            scheduler_class.step()
            scheduler_rej.step()
            s = np.random.choice(range(len(Ytr)),bsize, replace=False)
            h = mlp_class(Xtr[s]).squeeze()
            r=mlp_rej(Xtr[s]).squeeze()
          
            e = hinge(Ytr[s],h)
#          
            l1 = torch.max(alpha/2*(r+e),c*(1-beta*r))
        
            loss_r=torch.mean(torch.max(zero, l1))  
            L.append(loss_r)
#            1/0
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
        y_r=y_r.numpy().flatten()
        y_p2=y_p.numpy().flatten()
        Yts=Yts.numpy().flatten()

        ##############################################################
        #top 27 rejections
        
        import operator
        ss=[[k,v,w ]for k, v, w in sorted(zip(y_r, Yts, y_p2), key=operator.itemgetter(0))]
        ss2=np.array(ss)
        
        ss1=np.array(ss[27:])
        y_p3=ss1[:,2]

        a_r=accuracy(ss1[:,1],y_p3)
        print("accuracy with top 27 rejected=",a_r)

        m_c=accuracy(Yts, y_p2)
        print("accuracy without rejection=",m_c)
        acc_all.append(m_c)
        acc_rej.append(a_r)
print ('mean accuracy without rejection', np.mean(acc_all), '+/-', np.std(acc_all))
print ('mean accuracy with rejection', np.mean(acc_rej), '+/-', np.std(acc_rej))
        
