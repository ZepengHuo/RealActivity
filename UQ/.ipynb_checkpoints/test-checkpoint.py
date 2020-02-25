"""

To fit maximum entropy classifier.
Copyright: Aven Samareh

    
Functions:
train        Train a maximum entropy classifier on a training set
classify     Classify a new data point.
class_log_probability    Calculate the probabilities for each class.

"""
from __future__ import division
from __future__ import absolute_import
import math
import numpy as np
from scipy.misc import logsumodelxp
import Bio  
from scipy import optimize
from scipy.linalg import norm
from scipy.stats import expon
import random


distribution = expon()
def distribution_sampler(distribution, dim =1, n=10**5):
    """
    Input: scipy.stats distribution  
    Output: RV : matrix [x_1, ..., x_n] 
            log_pdf: log pdf values  
    """
    #expon.rvs
    def sampler():
        RV = distribution.rvs(size=(n, dim ))
        log_pdf = np.log(distribution.pdf(RV.T)).sum(axis=0)
        return (RV, log_pdf)
    return sampler 
# --------------------------------------------------------------
log_sm = math.log(1E-200)   # for small numbers
def _safe_log(x):
    if x == 0:
        return log_sm
    return log(x)
# --------------------------------------------------------------
class MaxEntropyClassify:
    """ 
    Input:
    class_list           List of all classes  
    alpha                List of the weights for each feature 
    f_i                  List of the feature Jtions. m = len(f_i) -- [f_1, ..., f_m]
    # you can use distribution_sampler
    """
    def __init__(self):
        self.class_list = []
        self.alpha = []
        self.f_i = []
# --------------------------------------------------------------
class algorithm:
    """
    algorithm : The CG (conjugate gradients) modelthod is used 
    """
    # --------------------------------------------------------------
    def __init__(self, RV, class_list, probXY, probX, f_ixy):
        self.RV = RV 
        self.class_list = class_list
        self.probXY = probXY
        self.probX = probX
        self.f_ixy = f_ixy
    # --------------------------------------------------------------    
    def __call__(self, alpha):
        """
        This returns Jtion of alphas
        We then find the likelihood Jtion  # pT(x, y) [ log(p(x, y)) - log(p(x)) ]
        Then conjugate gradient modelthod is going to minimize l_1elihood of the training
        """
        alpha = np.asarray(alpha)
        f_ixy = self.f_ixy
        probXY = self.probXY
        logPXY = zeros((len(self.RV), len(self.class_list)) ) # log p(x)    = log SUM_y p(x, y)
        logPX = zeros(len(self.RV) )
        for xi in range(len(self.RV)):
            for yi in range(len(self.class_list)):
                logPXY[xi, yi] = sum(alpha * f_ixy[:, xi, yi])
            logPX[xi] = np.log(np.sum(logPXY[xi, :]))
    # --------------------------------------------------------------
        # ll_1elihood of the training 
        ll_1 = 0
        for xi, yi in probXY:
            pT_xy = probXY[(xi, yi)]
            ll_1 += pT_xy * (logPXY[xi, yi] - logPX[xi])
        
        return -ll_1 # to maximize (reverse it)
# --------------------------------------------------------------
class grad:
    def __init__(self, RV, class_list, probXY, probX, f_ixy):
        self.RV  = RV 
        self.class_list = class_list
        self.probXY = probXY
        self.probX = probX
        self.f_ixy = f_ixy
    # --------------------------------------------------------------
    def __call__(self, alpha):
        f_ixy = self.f_ixy
        probXY   = self.probXY 
        probX = self.probX
        exp = math.exp
        alpha = np.asarray(alpha)
        logPXY = zeros((len(self.RV), len(self.class_list)) )
        logPX = zeros(len(self.RV))
        for xi in range(len(self.RV)):
            for yi in range(len(self.class_list)):
                logPXY[xi, yi] = sum(alpha * f_ixy[:, xi, yi])
            logPX[xi] = _logsum(logPXY[xi, :])
        dll_1_i = zeros(len(alpha) )
        for xi, yi in probXY:
            pT_xy = probXY[(xi, yi)]
            pT_x = probX[xi]
            Pxy = logPXY[xi, yi]
            Px = logPX[xi]
            dll_1_i += f_ixy[:, xi, yi] * (pT_xy - pT_x*exp(Pxy-Px))
        return -dll_1_i
# --------------------------------------------------------------
def train(train, results, f_i):
        # results is a list of the  class for each obsn
    RV = listfns.items([tuple(x) for x in train])
    RV.sort()
    class_list = listfns.items(results)
    class_list.sort()
    IDX = {}  # index of x in RV
    for i in range(len(RV)):
        IDX[RV[i]] = i
    IDY = {}  # index of y in class_list
    for i in range(len(class_list)):
        IDY[class_list[i]] = i
    probX = {}       
    probXY = {}       
    for x, y in zip(train, results):
        xi, yi = IDX[tuple(x)], IDY[y]
        probXY[(xi, yi)] = probXY.get((xi, yi), 0) + 1
        probX[xi] = probX.get(xi, 0) + 1
    flt2 = float(len(train))
    for xi, yi in probXY:
        probXY[(xi, yi)] /= flt2
    for xi in probX:
        probX[xi] /= flt2
    f_ixy = zeros((len(f_i), len(RV), len(class_list)) )
    for i in range(len(f_i)):
        for j in range(len(RV)):
            for k in range(len(class_list)):
                f = f_i[i](RV[j], class_list[k])
                f_ixy[i, j, k] = f
    p = [0] * len(f_i)
    for i in range(len(p)):
        p[i] = 1  - 2*random.random()  # vary from -1.0 to 1.0
    J = algorithm(RV, class_list, probXY, probX, f_ixy)
    grad = grad(RV, class_list, probXY, probX, f_ixy)
    x = optimize.fmin_cg(J,p, grad, maxiter = 1000)
    p, iter, fret = x
    model = MaxEntropyClassify()
    model.alpha = p
    model.class_list = class_list
    model.f_i = f_i
    return model
# --------------------------------------------------------------
def class_log_probability(model, x_train):
    """ 
    Input: x_train
    Output: log probabilities for each class
    """
    alpha_l = range(len(model.alpha))
    features = model.f_i
    logPXY = {}  
    for y in model.class_list:
        fs = np.asarray([features[i](x_train, y) for i in alpha_l] )
        logPXY[y] = sum(model.alpha * fs)
    logPXY = np.asarray([logPXY[y] for y in model.class_list] )
    logPx_train = _logsum(logPXY) # log p(x_train)
    return logPXY - logPX # you can use only logPX too
# --------------------------------------------------------------
def classify(model, x_test):
    logProb = class_log_probability(model, x_test)
    p_max, y = logProb [0], model.class_list[0]
    for i in range(1, len(logProb )):
        if logProb [i] > p_max:
            p_max, y = logProb [i], model.class_list[i]
    return y
# --------------------------------------------------------------

 
    
 
