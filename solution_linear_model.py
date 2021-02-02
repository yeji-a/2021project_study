
# coding: utf-8

# In[2]:


import numpy as np


class LinearRegression(object):
    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

        self._coef = None
        self._intercept = None
        self._new_X = None
    
    def fit(self,X,y):
        self._new_X = np.array(X)
        y = y.reshape(-1,1)
        
        if self.fit_intercept:
            intercept_vector = np.ones([len(self._new_X),1])
            self._new_X = np.concatenate((intercept_vector,self._new_X),axis=1)
            
        weights = np.linalg.inv(self._new_X.T.dot(self._new_X)).dot(self._new_X.T.dot(y)).flatten()
        
        if self.fit_intercept:
            self._intercept = weights[0]
            self._coef = weights[1:]
        else:
            self._coef = weights
    
    def predict(self,X):
        test_X = np.array(X)
        
        if self.fit_intercept:
            intercept_vector = np.ones([len(test_X),1])
            test_X = np.concatenate((intercept_vector,test_X),axis=1)
            
            weights = np.concatenate(([self._intercept],self._coef),axis=0)
        else:
            weights = self._coef
        return test_X.dot(weights)
    
    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept

