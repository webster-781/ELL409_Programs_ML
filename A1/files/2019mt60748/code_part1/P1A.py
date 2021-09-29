#!/usr/bin/env python
# coding: utf-8

# In[1]:


def get(phix,m):     # Returns phi only upto the no. of features 0 to m-1
    newphi = np.copy(phix.T)
    newphi = newphi[0:m]
    return newphi.T

def plot_all(x,hs,xl = 'X',yl = 'Y',legends = [''],xlog = False,scatterFirst = False, ylog = False):      # Plots all curves with x on X and all elements oh hs on Y, ll = xlabel, yl = ylabel, xlog = if x is on logarithmic scale, legends = legends
    plt.figure()
    plt.xlabel(xl)
    plt.ylabel(yl)
    if(xlog):
        plt.xscale('log')
    if(ylog):
        plt.yscale('log')
    if(scatterFirst):
        plt.scatter(x,hs[0],color = 'indigo')
    for h in hs[scatterFirst:]:
        plt.plot(x,h)
    plt.legend(legends)
    plt.show()

def make_batches(phix,tx,bs):                 # Returns the batches of phi,t with batch_size = bs
    N = phix.shape[0]
    newphi = np.copy(phix)
    newphi = np.concatenate((newphi,np.array([tx]).T),axis = 1)
    np.random.shuffle(newphi)
    sz = math.floor(N/bs)
    # print(sz,bs)
    newphi = newphi[0:sz*bs]
    N = phix.shape[0]
    newphi = np.array(np.split(newphi,sz))
    phis = []
    ts = []
    for ph in newphi:
        ts.append(ph.T[-1])
        phis.append((ph.T[0:-1]).T)
    return phis,ts

def MPPI(phi, t, m, lamda):
    phix = get(phi,m)
    wm = (( np.linalg.inv(lamda*np.eye(m) + (phix.T).dot(phix)).dot(phix.T)).dot(t))
    h = phix.dot(wm)
    return wm,h

def L1Error(h,t):   # h,t hypotheses and targets, the two vectors b/w which we need the norm
    return np.linalg.norm(h-t,1)

def LinfError(h,t): # h,t hypotheses and targets, the two vectors b/w which we need the norm
    return np.linalg.norm(h-t,np.inf)


def SSE(hi,ti):   # Returns Sum of Squares error between the hypothesis h and the target t
    N = hi.shape[0]
    cost = (1/N)*0.5*sum(np.square(hi-ti))
    return cost

def SSG(phii,ti,h,lamda,w):
    N = ti.shape[0]
    gd = np.dot(phii.T,h-ti)/N + lamda*w
    return gd

def batch_gd(phi, alpha, tx, iters, batch_size, m,lamda =0, errorf = SSE, gradient= SSG):
    phix = get(phi,m)
    w = np.ones(m,dtype = 'float')
    M = phix.shape[1]       # Number of features
    N = batch_size          # Number of data points
    for i in range(iters):
        phis,ts = make_batches(phix,tx,batch_size)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                for i in range(len(phis)):
                    phii = phis[i]
                    ti = ts[i]
                    h = phii.dot(w)

                    gd = gradient(phii,ti,h,lamda,w)

                    w = w - (alpha * gd)

                    h = phii.dot(w)
            except RuntimeWarning as e:
                return e
    h = phix.dot(w)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cost = errorf(h,tx)
        except RuntimeWarning as e:
            return e
    return w,cost,h

def check(phi,t,m):
    N = phi.shape[0]
    batch_sizes =[]
    alphas = []
    for bs in [1,2,5,10,20,50,100]:
        batch_sizes.append(bs)
        alpha,cost = find_alpha(phi,t,bs,1e-5,1,m)
        alphas.append(alpha,cost)
        print(bs,alphas[-1])
    plt.plot(batch_sizes,alphas)

def find_alpha(phi,t,batch_size,l,r,lamda,m,errorf):
    mid = 0
    cost = 0
    for _ in range(5):
        mid = (l+r)/2
        output = batch_gd(phi,mid,t,10000,batch_size,lamda = lamda,m = m)
        if(isinstance(output,RuntimeWarning)):
            r = mid
        elif(output[1] > 1e6):
            r = mid
        else:
            cost = output[1]
            l =mid
    return l,cost

def Cross_Validation(phi,tx,alpha,parts,lamda,m,batch_size = 1,iters=1000,errorf=SSE,gradient=SSG,pinv = True):   # Finds cross validation Error for the given arguments
    # print(m,lamda)
    phix = get(phi,m)
    N = phix.shape[0]
    bs = math.floor(N/parts)
    alls = np.array_split(range(N),parts)
    test_error = 0.0
    training_error = 0.0
    if(parts == 1):
        if(pinv):
            wi,hi = MPPI(phi,tx,m,lamda)   
            costi = SSE(hi,tx)      # Represents the training error
        else:
            wi,costi,hi = batch_gd(phi = phi, alpha = alpha,tx = tx,batch_size= batch_size,lamda = lamda, iters = iters,m = m)
        return costi
    else:
        for i in range(parts):
            phii = np.delete(phix,alls[i],axis = 0)
            ti = np.delete(tx,alls[i])
            if(pinv):
                wi,hi = MPPI(phii,ti,m,lamda)   
                hi = get(phii,m).dot(wi)
                costi = errorf(hi,ti)      # Represents the training error
            else:
                wi,costi,hi = batch_gd(phi = phii, alpha = alpha,tx = ti,batch_size= batch_size,lamda = lamda, iters = iters,m = m)
                hi = get(phii,m).dot(wi)
                costi = errorf(hi,ti)      # Represents the training error
            # Wi and Hi are weights and hypothesis values at all inputs in this training set
            hypothesis_i = phix[alls[i]].dot(wi)    # Represents the hypothesis values at the test set
            training_error += costi # Increment total training error
            test_error += errorf(hypothesis_i,tx[alls[i]]) # Increment total testing error
    test_error /= parts
    training_error /= parts
    print(test_error,training_error)
    return test_error,training_error


# In[2]:


import csv
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
M = 100 # Order of poly is M-1
from random import randint as rand
with open("gaussian.csv", 'r') as f:
    data = list(csv.reader(f,delimiter = ','))
# data= data[0:20]
data = np.array(data)
data = np.array([[float(d[0]),float(d[1])] for d in data])
data = data[np.argsort(data[:,0])]
x = np.array([d[0] for d in data], dtype = 'float')
t = np.array([d[1] for d in data], dtype= 'float')
phi = np.array([[pow(xv,i) for i in range(M)] for xv in x])
N = t.size


# In[28]:


errors = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 5, lamda = 0, m = i, pinv = False, iters = 100000) for i in range(2,10)]


# In[33]:


test,train = np.array(errors).T
plot_all(x = range(1,len(test)+1), hs = [test,train],  legends = ['Testing','Training'], xl ='Degree of Polynomial', yl ='Error',)


# In[49]:


errors = [Cross_Validation(phi,t,alpha = 1e-4, batch_size = 1,parts = 4, lamda = 0, m = i) for i in range(1,10)]


# In[4]:


arr = [pow(math.e,i) for i in range(-15,-3,2)]
errors = [Cross_Validation(phi,t,alpha = 0.5, batch_size = 5,parts = 4, lamda = i, m = 6, pinv =False) for i in arr]


# In[48]:


errors2 = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, errorf = SSE) for i in range(1,25)]
errors1 = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, errorf = L1Error) for i in range(1,25)]
errorsinf = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, errorf = LinfError) for i in range(1,25)]


# In[49]:


# Plots for size = 100

m = 10

s1,r1 = np.copy(np.array(errors1)).T
s2,r2 = np.copy(np.array(errors2)).T
s3,r3 = np.copy(np.array(errorsinf)).T
fig,axs = plt.subplots(1,2,figsize=(20,10))
axs[0].plot(range(2,m),s1[2:m])
axs[0].plot(range(2,m),r1[2:m])
axs[0].set(xlabel = 'Degree of Polynomial', ylabel = 'CVS')
axs[0].legend(['Testing','Training'])
axs[0].set_title('Using Error Function as L1 norm')

axs[1].plot(range(2,m),s3[2:m])
axs[1].plot(range(2,m),r3[2:m])
axs[1].set(xlabel = 'Degree of Polynomial', ylabel = 'CVS')
axs[1].legend(['Testing','Training'])
axs[1].set_title('Using Error Function as L infinity norm')

print(s1[0],s2[0],s3[0])

# plot_all(x = range(2,m),hs = [s2[2:m],r2[2:m]],xl = 'Degree of Polynomial',yl="CVS with L2",legends=['Testing','Training'])
# plot_all(x = range(2,m),hs = [s1[2:m],r1[2:m]],xl = 'Degree of Polynomial',yl="CVS with L1",legends=['Testing','Training'])
# plot_all(x = range(2,m),hs = [s3[2:m],r3[2:m]],xl = 'Degree of Polynomial',yl="CVS with Linf",legends=['Testing','Training'])

plot_all(x = range(m),hs = [s1[0:m],s2[0:m],s3[0:m]],xl = 'Degree of Polynomial',yl="CVS for error formulations",legends=['L1','L2','Linf'])

print([s1,s2,s3])


# In[51]:


error2 = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, errorf = SSE) for i in range(1,25)]
error1 = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, errorf = L1Error) for i in range(1,25)]
errorinf = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, errorf = LinfError) for i in range(1,25)]


# In[54]:


# Plots for size = 20
m = 11

xs1,xr1 = np.copy(np.array(error1)).T
xs2,xr2 = np.copy(np.array(error2)).T
xs3,xr3 = np.copy(np.array(errorinf)).T
fig,axs = plt.subplots(1,2,figsize=(20,10))

axs[0].plot(range(2,m),xs1[2:m])
axs[0].plot(range(2,m),xr1[2:m])
axs[0].set(xlabel = 'Degree of Polynomial', ylabel = 'CVS')
axs[0].legend(['Testing','Training'])
axs[0].set_title('Using Error Function as L1 norm')

axs[1].plot(range(2,m),xs3[2:m])
axs[1].plot(range(2,m),xr3[2:m])
axs[1].set(xlabel = 'Degree of Polynomial', ylabel = 'CVS')
axs[1].legend(['Testing','Training'])
axs[1].set_title('Using Error Function as L infinity norm')

print(xs1[0],xs3[0],s1[0],s3[0])

plot_all(x = range(m),hs = [s1[0:m],s2[0:m],s3[0:m]],xl = 'Degree of Polynomial',yl="CVS for error formulations",legends=['L1','L2','Linf'])


# In[105]:


# For size = 20
lamdas = [pow(10,i) for i in range(-10,0)]
errs = [Cross_Validation(phi,t,alpha = 1e-4, batch_size = 1, parts = 4, lamda = lamda, m = 6) for lamda in lamdas]


# In[106]:


m = 5
ts1,tr1 = np.copy(np.array(errs)).T
ts1 = ts1[0:m]
tr1 = tr1[0:m]
plot_all(x = lamdas[0:m],hs = [ts1,tr1],xl = 'Lamda',yl="CVS",legends=['Testing','Training'],xlog = True)


# In[118]:


# For size = 100
lamdas = [pow(10,i) for i in range(-20,0)]
errs = [Cross_Validation(phi,t,alpha = 1e-4, batch_size = 1, parts = 4, lamda = lamda, m = 6) for lamda in lamdas]


# In[122]:


m = 15
ts1,tr1 = np.copy(np.array(errs)).T
ts1 = ts1[0:m]
tr1 = tr1[0:m]
plot_all(x = lamdas[0:m],hs = [ts1,tr1],xl = 'Lamda',yl="CVS",legends=['Testing','Training'],xlog = True)


# In[21]:


# Size 20
# Solving using Gradient Descent: bs = 1
error2 = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, errorf = SSE, pinv = False) for i in range(1,15)]


# In[24]:


ts,tr = np.copy(np.array(error2)).T
m = 12
ts = ts[1:m]
tr = tr[1:m]
plot_all(x = range(1,m), hs = [ts,tr],xl = 'Order of Poylnomial', yl="CVS for SGD",legends=['Testing','Training'])


# In[13]:


# Size 20
# Solving using Gradient Descent: bs = 20
error3 = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 10, parts = 1, lamda = 0, m = i, errorf = SSE, pinv = False) for i in range(1,15)]


# In[5]:


ts,tr = np.copy(np.array(error3)).T
m = 12
ts = ts[1:m]
tr = tr[1:m]
plot_all(x = range(1,m), hs = [ts,tr],xl = 'Order of Poylnomial', yl="CVS for SGD",legends=['Testing','Training'])


# In[55]:


fig, axs = plt.subplots(1,2,figsize = [20,7])
it = np.argsort(data[0:20][:,0])
it2 = np.argsort(data[:,0])
axs[0].plot(x[it],t[it])
axs[0].set(xlabel = "X",ylabel="Y")
axs[1].set(xlabel = "X",ylabel="Y")
axs[0].set_title("Data Set Size: 20 ")
axs[1].set_title("Data Set Size: 100 ")
axs[1].plot(x[it2],t[it2])


# In[65]:


wx,hx = MPPI(phi,t,6,1e-7)
print(wx)
plot_all(x = x,hs = [t,hx],scatterFirst=True,xl='X',yl='Y',legends=['Hypothesis','Data'])
noise = hx -t
std = np.std(noise)
print(np.mean(noise))
print(np.std(noise))
print(np.std(noise)**2)
print()


# In[68]:


wx1,hx1 = MPPI(phi,t,6,1e-7)
print(wx1)
plot_all(x = x,hs = [t,hx1],scatterFirst=True,xl='X',yl='Y',legends=['Hypothesis','Data'])
noise1 = hx1 - t
std = np.std(noise1)
print(np.mean(noise1))
print(np.std(noise1))
print(np.std(noise1)**2)


# In[4]:


def gd_errors(phi, alpha, tx, iters, batch_size, m,lamda =0, errorf = SSE, gradient= SSG):
    phix = get(phi,m)
    errors = []
    w = np.ones(m,dtype = 'float')
    M = phix.shape[1]       # Number of features
    N = batch_size          # Number of data points
    for i in range(iters):
        phis,ts = make_batches(phix,tx,batch_size)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                for i in range(len(phis)):
                    phii = phis[i]
                    ti = ts[i]
                    h = phii.dot(w)

                    gd = gradient(phii,ti,h,lamda,w)

                    w = w - (alpha * gd)

                    h = phix.dot(w)
            except RuntimeWarning as e:
                return e
        errors.append(errorf(h,tx))
    h = phix.dot(w)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cost = errorf(h,tx)
        except RuntimeWarning as e:
            return e
    return w,cost,h,errors


# In[110]:


out1 = gd_errors(phi = phi, alpha = 1e-2, tx = t, iters = 10000, batch_size = 1, m = 6, lamda =0)
out2 = gd_errors(phi = phi, alpha = 1e-2, tx = t, iters = 10000, batch_size = 5, m = 6, lamda =0)
out3 = gd_errors(phi = phi, alpha = 1e-2, tx = t, iters = 10000, batch_size = 20, m = 6, lamda =0)


# In[115]:


fig,axs = plt.subplots(1,3,figsize=(20,5))
axs[0].plot(range(1,10001),out1[3],'green')
axs[1].plot(range(1,10001),out1[3],'blue')
axs[2].plot(range(1,10001),out1[3],'red')
axs[0].set(xlabel = 'Epochs',ylabel = 'Error of Hypothesis')
axs[1].set(xlabel = 'Epochs',ylabel = 'Error of Hypothesis')
axs[2].set(xlabel = 'Epochs',ylabel = 'Error of Hypothesis')
axs[0].set_title("Batch Size 1")
axs[1].set_title("Batch Size 5")
axs[2].set_title("Batch Size 20")
axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[2].set_xscale('log')


# In[5]:


out1 = gd_errors(phi = phi, alpha = 1e-2, tx = t, iters = 1000, batch_size = 1, m = 6, lamda =0)
out2 = gd_errors(phi = phi, alpha = 1e-2, tx = t, iters = 1000, batch_size = 20, m = 6, lamda =0)
out3 = gd_errors(phi = phi, alpha = 1e-2, tx = t, iters = 1000, batch_size = 100, m = 6, lamda =0)


# In[7]:


fig,axs = plt.subplots(1,3,figsize=(20,5))
axs[0].plot(range(1,1001),out1[3],'green')
axs[1].plot(range(1,1001),out1[3],'blue')
axs[2].plot(range(1,1001),out1[3],'red')
axs[0].set(xlabel = 'Epochs',ylabel = 'Error of Hypothesis')
axs[1].set(xlabel = 'Epochs',ylabel = 'Error of Hypothesis')
axs[2].set(xlabel = 'Epochs',ylabel = 'Error of Hypothesis')
axs[0].set_title("Batch Size 1")
axs[1].set_title("Batch Size 20")
axs[2].set_title("Batch Size 100")
axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[2].set_xscale('log')

