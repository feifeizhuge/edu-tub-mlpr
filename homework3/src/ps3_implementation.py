""" ps3_implementation.py

PUT YOUR NAME HERE:
Till Rohrmann, Matrikelnummer: 343756


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
import sys
from mpl_toolkits.mplot3d import Axes3D

def zero_one_loss(y_true, y_pred):
    ''' This function calculates the fraction of wrongly labeled data
    
        Input:
            y_true: n vector containing the right labels
            y_pred: n vector containing the predicted labels
            
        Output:
            loss: percentage of wrongly labeled data points
            
        Author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    assert y_true.size == y_pred.size, 'Size of y_true and y_pred has to be equal.'
    
    y_pred = 2 * (y_pred >= 0) - 1;
    
    return np.sum(y_true != y_pred) / float(y_true.size);

def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' This function performs a cross validation in order to estimate the best parameters
        from a given range. For this purpose it separates the data randomly into nfolds parts 
        and uses each of these parts as the testing data set and the rest for the training data set.
        For each testing data set the loss is calculated and sumed up. The whole procedure is repeated
        nrepetitions times, each time with a shuffeling the input data.
        
        Input:
            X : d x n data vector containing the data points in its columns
            y : 1 x n data vector containing the to the data points associated labels/function values
            method : class implementing a fit and predict method
            params : list of parameter and parameter values pair from which the different 
                candidates for the cross validation are selected
            loss_function : loss_function which measures the goodness of the found solution
            nfolds : number of folds to use for the cross validation
            nrepetitions : number of repetitions
            
        Output:
            method : method object on which the cross validation was applied\
            
        Author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    error = 0;
    n = X.shape[1];
    foldSize = float(n) / nfolds;
    parametervalues = params[1::2];
    
    minError = np.inf;
    
    candidates = it.product(*parametervalues);
    startTime = time.time();
    numCandidates = reduce(lambda x, y: x * len(y), parametervalues, 1); 
    totalNumberFolds = numCandidates * nrepetitions * nfolds;
    foldCounter = 0;
    currentFoldCounter = 0;
    maxStepsTimeEstimation = 60;
    minCandidate = False;
    
    for icand, candidate in enumerate(candidates):
        error = 0;
        for irep, _ in enumerate(np.arange(nrepetitions)):
            indices = np.arange(n);
            rng = np.random.RandomState();
            rng.shuffle(indices);
            
            exceptionRaised = False;
            
            for i, fold in enumerate(np.arange(nfolds)):
                timePerFold = np.inf if currentFoldCounter == 0 else (float(time.time() - startTime)) / currentFoldCounter;
                estimatedTime = (totalNumberFolds - foldCounter) * timePerFold;
                print('Start (' + str(icand) + '/' + str(numCandidates) + ')th candidate, (' + 
                      str(irep) + '/' + str(nrepetitions) + ')th repetition, (' + 
                      str(i) + '/' + str(nfolds) + ')th fold. ' + ('%.1f' % estimatedTime) + 's until completion.');
                startIndex = np.floor(fold * foldSize);
                endIndex = np.floor((fold + 1) * foldSize);
                
                testX = X[:, indices[startIndex:endIndex]];
                testY = y[:, indices[startIndex:endIndex]];
                
                trainingX = X[:, np.hstack((indices[:startIndex], indices[endIndex:]))];
                trainingY = y[:, np.hstack((indices[:startIndex], indices[endIndex:]))];
                
                try:
                    method.fit(trainingX, trainingY, *candidate);
                    method.predict(testX);
                    error += loss_function(testY, method.ypred);
                except Exception as ex:
                    print(ex);
                    error = np.inf;
                    exceptionRaised = True;
                    break;
                
                foldCounter += 1;
                currentFoldCounter += 1;
                
                # always take the last maxStepsTimeEstimation/2 for the time extrapolation
                if(currentFoldCounter == maxStepsTimeEstimation / 2):
                    nextStartTime = time.time();
                    
                if(currentFoldCounter >= maxStepsTimeEstimation):
                    currentFoldCounter = maxStepsTimeEstimation / 2;
                    startTime = nextStartTime;
                    nextStartTime = time.time();
            
            if exceptionRaised:
                break;
        error /= (nrepetitions * nfolds);
        
        if(np.ndim(error) == 0):
            condition = error < minError
        else:
            condition = all(error.flat < minError)
        
        if condition:
            minError = error;
            minCandidate = candidate;
    
    if(minCandidate is False):
        raise Exception('Could not find working parameter tuple.');
    
    # fit the method to all data at once
    method.fit(X, y, *minCandidate);
    method.cvloss = minError;
        
    return method

    
class krr():
    ''' This class implements the functionality to perform a kernel ridge regression
        
        Input:
            kernel : string denoting the kernel which shall be used
                    values: 'linear', 'polynomial', 'gaussian'
            kernelparameter : optional kernel parameter
            regularization : Regularization constant for the krr
        
        Author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
        self.trainingData = np.array([]);
        self.alpha = np.array([]); 
        self.fitted = False;
        self.ypred = np.array([]);
        self.cvloss = -1;
    
    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' This function applies the kernel ridge regression on the data set
            in order to fit it as good as possible.
            
            Input:
                X : d x n training data set with the columns being the data points
                y : 1 x n vector containing the labels/function values
                kernel : string containing the kernel function to be used
                kernelparameter : optional kernel parameter for the kernel function
                regularization : regularization constant
                
            Output:
                self : return itself after fitting to data
                
            Author:
                Till Rohrmann, till.rohrmann@campus.tu-berlin.de
        '''
        
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization
            
        K = self.kernelFunction(X, X);
        self.trainingData = X;
        n = X.shape[1];
            
        if(self.regularization == 0):
            # efficient leave one out cv
            epsilon = 1e-10;
            eigenvalues, eigenvectors = np.linalg.eigh(K);
            
            # set the candidates for the regularization constant lie between
            # the minimum and maximum eigenvalue of matrix K. The minimal
            # eigenvalue is bounded below by epsilon. Otherwise there might
            # cases when C is too small and the regularized matrix K cannot
            # be inverted.
            candidates = np.logspace(np.log10(np.max([np.min(eigenvalues), epsilon])), 
                                     np.log10(np.max(eigenvalues)), base=10);
            UY = np.dot(eigenvectors.T, y.T);
            minError = np.inf;
           
            for candidate in candidates:                   
                LiLC = eigenvalues / (eigenvalues + candidate);
                ULiLC = eigenvectors * LiLC[np.newaxis, :];
                sdiag = np.sum((ULiLC) * eigenvectors, axis=1);
                
                # check whether K(K+CI)^-1 is different from the identity matrix
                # if not then this means that C has been choosen too small
                if(np.sum(sdiag == 1) > 0):
                    continue;
                
                prediction = np.dot(ULiLC, UY);
                 
                error = np.sum(((y.T - prediction) / (1 - sdiag)) ** 2) / n;
                 
                if(error < minError):
                    minError = error;
                    self.alpha = np.dot(eigenvectors, UY / (eigenvalues + candidate)[:, np.newaxis]);
                    self.regularization = candidate;
                    self.fitted = True;
            
            if self.fitted == False:
                raise Exception('Could not find regularization constant for LOOCV with kernel:' + 
                                self.kernel + ' kernelparameter:' + str(self.kernelparameter));
        else:
            A = (K + self.regularization * np.identity(n));
            
            # check the condition of A. If it is too high, then A is singular
            # and cannot be inverted
            if np.linalg.cond(A) < 1 / sys.float_info.epsilon:
                self.alpha = np.linalg.solve(A, y.T);
                self.fitted = True;
            else:
                self.fitted = False;
                self.alpha = np.array([]);
                raise Exception('Matrix A is singular: regularization:' + str(self.regularization) + 
                                ' kernel:' + self.kernel + ' kernelparam:' + str(self.kernelparameter))
            
        return self

    def kernelFunction(self, x, y):
        ''' This function calculates the kernel function for the arguments x and y
            depending on the selected kernel function.
        
            Input:
                x : d times n data array containing the data points in its columns
                y : d times m data array containing the data points in its columns
                
            Output:
                K : n times m kernel matrix where K[i,j] = kernelFunction(x[:,i],y[:,j])
                
            Author:
                Till Rohrmann, till.rohrmann@campus.tu-berlin.de
        '''
        if(self.kernel == 'linear'):
            K = np.dot(x.T, y);
        elif(self.kernel == 'polynomial'):
            K = (np.dot(x.T, y) + 1) ** self.kernelparameter;
        elif(self.kernel == 'gaussian'):
            K = np.exp(-(np.sum((x[:, :, np.newaxis] - 
                                 y[:, np.newaxis, :]) ** 2, axis=0)) / (2 * self.kernelparameter ** 2));
        else:
            raise 'Invalid value for kernel function:' + self.kernel;
        
        return K;
                    
    def predict(self, X):
        ''' This function predicts the labels/function values of the data X based on a preceding fitting phase.
            The prediction is saved in y_pred.
        
            Input:
                X : d times n data array with the columns being the data points
                
            Output:
                self : returns itself
                
            Author:
                Till Rohrmann, till.rohrmann@campus.tu-berlin.de
        '''
        K = self.kernelFunction(X, self.trainingData);
        self.ypred = np.dot(K, self.alpha).T;
        
        return self
    


