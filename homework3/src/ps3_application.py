""" ps3_application.py

PUT YOUR NAME HERE:
Till Rohrmann, Matrikelnummer: 343756


Write the functions
- roc_curve
- apply_krr

Write your code in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import pylab as pl
# import matplotlib as pl
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.lines import Line2D
from scipy.stats import norm
import os
import pickle
import re;
import time;


import ps3_implementation as imp
imp = reload(imp)


def roc_curve(n):
    ''' This function draws the empirical and analytical ROC curve for linear classifier of a 
        mixture Gaussian distribution. p(y=1) = p(y=-1) = 0.5 and
        p(x|y=1) ~ N(2,1) and p(x|y=-1) ~ N(0,1).
        
        Input:
            n : sample size for the empirical ROC curve estimate
            
        Author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    # sampling from the mixture distribution
    rng = np.random.RandomState()
    y = 2 * (rng.rand(n) >= 0.5) - 1;
    x = rng.randn(n) + (y > 0) * 2;
    
    numClassA = np.sum(y == -1);
    numClassB = np.sum(y == 1);
    
    resolution = 1000;
    
    xspacing = np.linspace(-5, 7, resolution);
    
    # class y=1 are the outliers
    
    classification = x[:, np.newaxis] < xspacing[np.newaxis, :];
    
    empFalsePositiveRate = classification[y == 1, :].sum(axis=0) / float(numClassB);
    empTruePositiveRate = classification[y == -1, :].sum(axis=0) / float(numClassA);
    
    anaFalsePositiveRate = norm.cdf(xspacing - 2);
    anaTruePositiveRate = norm.cdf(xspacing);
    
    
    pl.figure();
    pl.plot(empFalsePositiveRate, empTruePositiveRate);
    pl.plot(anaFalsePositiveRate, anaTruePositiveRate);
    pl.legend(['Empirical ROC', 'Analytical ROC']);
    pl.title('ROC-Curve(' + str(n) + ')');
    pl.ylabel('True positive rate');
    pl.xlabel('False positive rate');


def roc_fun(y_true, y_pred):
    ''' This function calculates the TPR and FPR for a given prediction
        depending on the bias.
        
        Input:
            y_true : 1 x n vector containing the right labels
            y_pred : 1 x n vector containing the predicted labels
            
        Output:
            result : 2 x resolution matrix containing in its first row the TPR
                and its second the FPR. The number of columns depends on the
                resolution of the bias chosen.
            
        Author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    numClassA = sum(y_true.flatten() == -1);
    numClassB = sum(y_true.flatten() == 1);
    # important that this is odd because otherwise the bias
    # b=0 won't be used. If the prediction values are really small,
    # then every bias b != 0 causes the TPR and FPR to be either 0
    # or 1.
    resolution = 1001;
    bias = np.linspace(-2, 2, resolution);
    
    classification = (y_pred.T - bias[np.newaxis, :]) < 0;
    
    truePositiveRate = classification[y_true.flatten() == -1, :].sum(axis=0) / float(numClassA);
    falsePositiveRate = classification[y_true.flatten() == 1, :].sum(axis=0) / float(numClassB);
    
    return np.array([truePositiveRate, falsePositiveRate]);

    
def apply_krr(reg=False):
    ''' This function applies the krr to the provided data set in order
        to find a good classification. The results are stored in a dictionary
        which is at the end pickled.
        
        Usage:
            It is important to adapt the path of the datasets.
        
        Input:
            reg : boolean variable indicating whether the regularization constant
                shall be estimated by LOOCV or drawn from a provided range. True
                means that the provided range is used and False means that the
                LOOCV will be used.
                
        Author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    # IMPORTANT: Adapt path to where the data sets have been stored
    path = 'ps3_datasets';
    testSuffix = 'xtest';
    trainXSuffix = 'xtrain';
    trainYSuffix = 'ytrain';
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))];
    
    datasetNames = set();
    
    for filename in files:
        m = re.search('U04_([^\.]*)\.dat', filename);
        if m != None:
            datasetNames.add(m.group(1)[:m.group(1).rfind('-')]);
            
    result = {};
    
    if reg:
        regularization = np.logspace(-2, 2, 10);
    else:
        regularization = [0];
        
    gaussianKernelParams = np.logspace(-2, 2, 10);
    polynomialKernelParams = np.arange(1, 10);
    
    nrep = 5;
    nfs = 10;
            
    for dataset in datasetNames:
        print('Dataset: ' + dataset);
        # training phase
        filenameX = 'U04_' + dataset + '-' + trainXSuffix + '.dat';
        filenameY = 'U04_' + dataset + '-' + trainYSuffix + '.dat';
        filenameTestX = 'U04_' + dataset + '-' + testSuffix + '.dat';
        X = np.loadtxt(os.path.join(path, filenameX), dtype=float);
        Y = np.loadtxt(os.path.join(path, filenameY), dtype=float)[np.newaxis, :];
        testX = np.loadtxt(os.path.join(path, filenameTestX), dtype=float);
        
        print('Shape: ' + str(X.shape));
        
        # linear cv
        startTime = time.time();
        krrLinear = imp.krr();
        linearParams = ['kernel', ['linear'], 'kernelparam', [0], 'regularization', regularization];
        imp.cv(X, Y, krrLinear, linearParams, nrepetitions=nrep, nfolds=nfs);
        timeLinear = time.time() - startTime;
        
        # polynomial cv
        startTime = time.time();
        krrPolynomial = imp.krr();
        polynomialParams = ['kernel', ['polynomial'], 'kernelparam', 
                            polynomialKernelParams, 'regularization', regularization];
        imp.cv(X, Y, krrPolynomial, polynomialParams, nrepetitions=nrep, nfolds=nfs);
        timePolynomial = time.time() - startTime;
        
        # gaussian cv
        startTime = time.time();
        krrGaussian = imp.krr();
        gaussianParams = ['kernel', ['gaussian'], 'kernelparam', 
                          gaussianKernelParams, 'regularization', regularization];
        imp.cv(X, Y, krrGaussian, gaussianParams, nrepetitions=nrep, nfolds=nfs);
        timeGaussian = time.time() - startTime;
        
        krr = [krrLinear, krrPolynomial, krrGaussian][np.argmin([krrLinear.cvloss, 
                                                                 krrPolynomial.cvloss, krrGaussian.cvloss])];
        minTime = [timeLinear, timePolynomial, timeGaussian][np.argmin([krrLinear.cvloss, 
                                                                        krrPolynomial.cvloss, krrGaussian.cvloss])];
        
        krr.predict(testX);
        
        dictionary = dict();
        dictionary['kernel'] = krr.kernel;
        dictionary['kernelparameter'] = krr.kernelparameter;
        dictionary['regularization'] = krr.regularization;
        dictionary['cvloss'] = krr.cvloss;
        dictionary['ypred'] = krr.ypred;
        
        result[dataset] = dictionary;
        
        # plot ROC curve and calculate AUC
        params = ['kernel', [krr.kernel], 'kernelparam', [krr.kernelparameter], 
                  'regularization', [krr.regularization]];
        rocKRR = imp.krr();
        imp.cv(X, Y, rocKRR, params, loss_function=roc_fun, nrepetitions=nrep, nfolds=nfs);
        
        truePositiveRate = rocKRR.cvloss[0];
        falsePositiveRate = rocKRR.cvloss[1];
        
        # Simpson rule for integration
        xdiff = falsePositiveRate[1:] - falsePositiveRate[:-1];
        ysum = (truePositiveRate[1:] + truePositiveRate[:-1]) / 2
        AUC = np.dot(ysum, xdiff);
    
        pl.figure();
        pl.plot(falsePositiveRate, truePositiveRate);
        pl.ylabel('True positive rate');
        pl.xlabel('False positive rate');

        if reg == True:
            pl.title('ROC-Curve Dataset:' + dataset + ' AUC=' + ('%.3f' % AUC) + 
                     ' cvloss:' + ('%.3f' % (dictionary['cvloss'])) + ' time:' + str('%.1f' % minTime) + 's' + 
                  '\n Kernel:' + dictionary['kernel'] + ' parameter:' + ('%.3f' % dictionary['kernelparameter']) + 
                  ' regularization:' + ('%.3f' % dictionary['regularization']));
        else:
            pl.title('LOOCV ROC-Curve Dataset:' + dataset + ' AUC=' + ('%.3f' % AUC) + 
                     ' cvloss:' + ('%.3f' % (dictionary['cvloss'])) + ' time:' + str('%.1f' % minTime) + 's' + 
                  '\n Kernel:' + dictionary['kernel'] + ' parameter:' + ('%.3f' % dictionary['kernelparameter']) + 
                  ' regularization:' + ('%.3f' % dictionary['regularization']));
        
        print('Dataset:' + dataset + ' kernel:' + dictionary['kernel'] + ' cvloss:' + 
              str(dictionary['cvloss']) + ' AUC:' + str(AUC) + ' time:' + ('%.1f' % minTime));
        
    if reg:
        filename = 'results.p'
    else:
        filename = 'resultsLOOCV.p'
        
    pickle.dump(result, open(filename, 'wb'));
    
    
if __name__ == '__main__':
    
    values = np.array([25, 50, 100, 500, 1000, 10000]);
    for n in values:
        roc_curve(n);
    apply_krr(True);
    apply_krr(False);
    pl.show();
