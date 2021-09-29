#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import warnings
df = pd.read_csv('train.csv')
[df['month'],df['day'],df['year']] = [[float(df['id'][i].split('/')[j]) for i in range(df.shape[0])] for j in [0,1,2]]
M = 10
df = df.sort_values(by =['year','month'])


# In[2]:


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

def Cross_Validation(phi,tx,alpha,parts,lamda,m,batch_size = 1,iters=5000,errorf=SSE,gradient=SSG,pinv = True):   # Finds cross validation Error for the given arguments
    # print(m,lamda)
    phix = get(phi,m)
    N = phix.shape[0]
    bs = math.floor(N/parts)
    alls = np.array_split(range(N),parts)
    test_error = 0.0
    training_error = 0.0
    for i in range(parts):
        phii = np.delete(phix,alls[i],axis = 0)
        ti = np.delete(tx,alls[i])
        if(pinv):
            wi,hi = MPPI(phii,ti,m,lamda)   
            costi = SSE(hi,ti)      # Represents the training error
        else:
            wi,costi,hi = batch_gd(phi = phii, alpha = alpha,tx = ti,batch_size= batch_size,lamda = lamda, iters = iters,m = m)
        # Wi and Hi are weights and hypothesis values at all inputs in this training set
        hypothesis_i = phix[alls[i]].dot(wi)    # Represents the hypothesis values at the test set
        training_error += costi # Increment total training error
        test_error += SSE(hypothesis_i,tx[alls[i]]) # Increment total testing error
    test_error /= parts
    training_error /= parts
    test_error = math.sqrt(test_error)
    training_error = math.sqrt(training_error)
    print(test_error,training_error)
    return test_error,training_error


# In[3]:


x1 = np.array(df['year'])
x2 = np.array(df['month'])
x1 = np.array([x1[i] - 5 for i in range(len(x1))])
x2 = np.array([x2[i] - 1 for i in range(len(x2))])
t = np.array(df['value'])
x3 = np.array([x1[i]*12 + x2[i] for i in range(x1.shape[0])])
phi = np.array([[pow(xv,i) for i in range(M)] for xv in x3])
months= []
years = []
for j in range(0,12):
    month = []
    for i in range(x1.shape[0]):
        if(x2[i] == j):
            month.append(i)
    months.append(month)

qtrs= []
for j in range(1,5):
    qtr = []
    for i in range(x1.shape[0]):
        if(x2[i] <= j*4 and x2[i] > (j-1)*4):
            qtr.append(i)
    qtrs.append(qtr)

for j in range(0,11):
    qtr = []
    year = []
    for i in range(x1.shape[0]):
        if(x1[i] == j):
            year.append(i)
    years.append(year)
print(phi.shape)


# In[15]:


plt.figure(figsize = (10,5))
plt.xlabel("Time")
plt.ylabel("Observed Values")
plt.xlim(-10,180)
plt.plot(x3[months[0]],t[months[0]],'green') # 9
plt.plot(x3[months[1]],t[months[1]],'deepskyblue') # 7
plt.plot(x3[months[2]],t[months[2]],'pink') # 5
plt.plot(x3[months[3]],t[months[3]],'orange') # 3
plt.plot(x3[months[4]],t[months[4]],'teal') # 1
plt.plot(x3[months[5]],t[months[5]],'magenta') # 2
plt.plot(x3[months[6]],t[months[6]],'blue') # 4
plt.plot(x3[months[7]],t[months[7]],'yellow') # 6
plt.plot(x3[months[8]],t[months[8]],'black') # 8
plt.plot(x3[months[9]],t[months[9]],'hotpink') # 10
plt.plot(x3[months[10]],t[months[10]],'brown') # 12
plt.plot(x3[months[11]],t[months[11]],'violet') # 11
plt.legend([f"Month : {i}" for i in range(1,13)])
plt.show()


# In[7]:


# m3456 = months[3] + months[4] + months[5] + months[6]
# sort(m3456)
plt.figure(figsize=(10,5))
plt.xlabel("Time (x3)")
plt.ylabel("Observed Values")
plt.scatter(x3,t)
plt.plot(x3,t)


# In[9]:


errors = [Cross_Validation(phi,t, alpha = 1e-4, batch_size = 1, parts = 10, lamda = 0, m = i, pinv = False) for i in range(2,25)]


# In[16]:


weights = []
hyps = []
for i in range(12):
    weight,hyp = MPPI(phi[months[i]],t[months[i]],8,1e-4)
    # print(i)
    # weight,costi,hyp = batch_gd(phi = phi[months[i]],tx = t[months[i]], m = 3, alpha = 1e-4, lamda = 0, iters= 5000)
    weights.append(weight)
    hyps.append(hyp)
weights = np.array(weights)
hyps = np.array(hyps)


# In[17]:


m1 = 6
test_all = np.zeros(m1-1)
train_all = np.zeros(m1-1)
for mt in range(12):
    # print('Month : ',mt)
    sz = len(months[mt])
    errors = [Cross_Validation(phi[months[mt]],t[months[mt]], alpha = 1e-4, batch_size = 1, parts = sz, lamda = 1e-4, m = i, pinv = True) for i in range(1,m1)]
    test,train = np.copy(np.array(errors)).T
    # plot_all(x = range(m1-1), hs = [test,train],  legends = ['Testing','Training'], xl ='Degree of Polynomial', yl ='Error',)
    test_all += test
    train_all += train
test_all /= 12
train_all /= 12
print("Cumulative Errors")
# plot_all(x = range(m1-1), hs = [test_all,train_all],  legends = ['Testing','Training'], xl ='Degree of Polynomial', yl ='Error',)


# In[45]:


degrees = [2,0,1,3,2,0,4,1,2,3,2,2]
ms = [deg + 1 for deg in degrees]
ws = []
hys = []
fig,axs = plt.subplots(4,3,figsize=(22,20))

for i in range(12):
    print("Month: ",i+1)
    wt,hypt = MPPI(phi[months[i]],t[months[i]],ms[i],1e-4)
    print(wt)
    # errs = [Cross_Validation(phi[months[i]],t[months[i]], alpha = 1e-4, batch_size = 1, parts = sz, lamda = 0, m = j, pinv = True) for j in range(1,7)]
    # tr,ts = np.array(errs).T
    # r = int(i/3)
    # c = i%3
    # print(r,c)
    # axs[r,c].plot(range(0,6),tr)
    # axs[r,c].plot(range(0,6),ts)
    # axs[r,c].set_title(f"Month {i+1}")
    # axs[r,c].set(xlabel ='Degree of Polynomial',ylabel='CVS')

    # axs[r,c].legend(["Testing","Training"])
    ws.append(wt)
    # hy = get(phi[months[i]],ms[i]).dot(wt)

# fig.show()

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')
#     hys.append(hy)


# In[47]:


print(ws)
ans2 = []
df1 = pd.read_csv('test.csv')
[df1['month'],df1['day'],df1['year']] = [[float(df1['id'][i].split('/')[j]) for i in range(df1.shape[0])] for j in [0,1,2]]
print(weights[0])
for i in range(10):
    mth = int(df1['month'][i]) - 1
    year = int(df1['year'][i]) - 5
    tim = mth + year*12
    phit = np.array([pow(tim,i) for i in range(0,ms[mth])])
    print(df1['id'][i],end= '')
    print(',',end = '')
    print(phit.dot(ws[mth]))
    ans2.append(phit.dot(ws[mth]))


# In[20]:


# Cross Validating to find best values of lamda for each curve
lamda_high = 15
lamdas1 = [pow(10,i) for i in range(-15,lamda_high)]
test_all = np.zeros(lamda_high +15)
train_all = np.zeros(lamda_high+15)
for mt in range(12):
    # print('Month : ',mt)
    sz = len(months[mt])
    errors = [Cross_Validation(phi[months[mt]],t[months[mt]], alpha = 1e-4, batch_size = 1, parts = sz, lamda = i, m = ms[mt], pinv = True) for i in lamdas1]
    test,train = np.copy(np.array(errors)).T
    # plot_all(x = lamdas1, xlog = True, hs = [test,train],  legends = ['Testing','Training'], xl ='Lamda', yl ='Error',)
    test_all += test
    train_all += train
test_all /= 12
train_all /= 12
# print("Cumulative Errors")
# plot_all(lamdas1, hs = [test_all,train_all],  legends = ['Testing','Training'], xl ='Lamda', yl ='Error',)


# In[40]:


lams = [pow(10,i) for i in [-2,-2,-1,-3,-2,-2,-3,-1,2,-2,-2,-2]]


# In[41]:


ws1 = []
hys1 = []
for i in range(12):
    wt,hy = MPPI(phi[months[i]],t[months[i]],ms[i],lams[i])
    # print(i)
    # weight,costi,hyp = batch_gd(phi = phi[months[i]],tx = t[months[i]], m = 3, alpha = 1e-4, lamda = 0, iters= 5000)
    ws1.append(wt)
    hys1.append(hy)
weights = np.array(weights)
hyps = np.array(hyps)


# In[16]:


mh = 11
plt.scatter(x3[months[mh]],t[months[mh]])
plt.plot(x3[months[mh]],hys[mh],'green')


# In[43]:


# Submit in the morning
ans3 = []
df1 = pd.read_csv('test.csv')
[df1['month'],df1['day'],df1['year']] = [[float(df1['id'][i].split('/')[j]) for i in range(df1.shape[0])] for j in [0,1,2]]
print(weights[0])
for i in range(10):
    mth = int(df1['month'][i]) - 1
    year = int(df1['year'][i]) - 5
    tim = mth + year*12
    phit = np.array([pow(tim,i) for i in range(0,ms[mth])])
    print(df1['id'][i],end= '')
    print(',',end = '')
    print(phit.dot(ws1[mth]))
    ans3.append(phit.dot(ws1[mth]))


# In[94]:


ans = 100*np.ones(10)
an2 = np.copy(ans2)
an3 = np.copy(ans3)
print(np.sum(np.square(ans-an3)))
print(20*SSE(ans,an2))
print(20*SSE(an2,an3))


# |correct - 100|*2 = 7507
# 
# |ans2 - 100|*2 = 7594
# 
# (ans2 - 100)2 - (c-100)2 = (7594 - 7507)*2 = 174
# (105.3)2 - 174 = (c-100)2
# c-100 = -104.47
# c = -4.47
# 

# In[1]:





# 
