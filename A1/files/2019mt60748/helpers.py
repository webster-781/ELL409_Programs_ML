import csv
import math
import numpy as np
import warnings
from random import randint as rand

def get(phix,m):     # Returns phi only upto the no. of features 0 to m-1
    newphi = np.copy(phix.T)
    newphi = newphi[0:m]
    return newphi.T

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

def MPPI(phi, t, m, lamda):         # Returns Moore Penrose Pseudo Inverse solution
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

def SSG(phii,ti,h,lamda,w): # Returns the gradient of h wrt t
    N = ti.shape[0]
    gd = np.dot(phii.T,h-ti)/N + lamda*w
    return gd

def batch_gd(phi, alpha, tx, iters, batch_size, m,lamda =0, errorf = SSE, gradient= SSG):   # Performs batch gradient descent
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


def find_alpha(phi,t,batch_size,l,r,lamda,m,errorf):    # Finds optimal learning rate alpha using binary search
    mid = 0
    cost = -1
    for _ in range(5):
        mid = (l+r)/2
        output = batch_gd(phi,mid,t,2500,batch_size,lamda = lamda,m = m)
        if(isinstance(output,RuntimeWarning)):
            r = mid
        elif(output[1] > 1e6):
            r = mid
        else:
            cost = output[1]
            l =mid
    if(cost == -1):
        output = batch_gd(phi,1e-6,t,5000,batch_size,lamda = lamda,m = m)
    return output

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
