""" ps2_implementation.py

PUT YOUR NAME HERE:
Till Rohrmann, Matrikelnummer: 343756


Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""

from __future__ import print_function
from __future__ import division
import numpy as np
from scipy.spatial.distance import cdist # fast distance matrices
from scipy.cluster.hierarchy import dendrogram # you can use this
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import warnings;

warnings.filterwarnings('ignore', 'invalid value encountered in add');

def kmeans(X, k, max_iter=100):
    """ Performs k-means clustering

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations

    Output:
    mu: (d x k) matrix with each cluster center in one column
    r: assignment vector
    
    Description:
        This function performs a k-means cluster analysis on the given dataset X.
        It tries to find k clusters and terminates either if the maximum number of
        iterations has been reached or if the cluster membership does not change
        anymore.
        
    Author:
        Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    """
    
    n = X.shape[1];
    
    # select initial clusters
    clusterIndices = np.random.permutation(np.arange(n))[0:k];
    
    mu = X[:,clusterIndices].copy();
    r = np.ones(n)*-1;
    
    for i in range(max_iter):
        distance = np.sqrt(((X[:,:,np.newaxis] - mu[:,np.newaxis,:])**2).sum(axis=0));
        rold = r;
        r = np.argmin(distance,axis=1);
        
#         plt.scatter(X[0,:],X[1,:],23,r);
#         plt.show();
        
        for j in range(k):
            mu[:,j] = X[:,r==j].sum(axis=1)/(r==j).sum();
         
        numChanges = (r != rold).sum();
        
        if(numChanges == 0):
            break;
        
        lossValue = np.sqrt(((X - mu[:,r])**2).sum(axis=0)).sum()
        
        print('Iteration:' + str(i) + ' #changes:' + str(numChanges) + ' loss function value:' + str(lossValue));
        
    
    return mu,r;

def kmeans_crit(X, r):
    """ Computes k-means criterion
    
    Input: 
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector
    
    Output:
    value: scalar for sum of euclidean distances to cluster centers
    """
    k = max(r)+1;
    result = 0;
    for j in range(k):
        if (r==j).sum() > 0:
            m = X[:,r==j].sum(axis=1)/(r==j).sum();
            result += np.sqrt(((X[:,r==j]-m[:,np.newaxis])**2).sum(axis=0)).sum();
    
    return result;


def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector

    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    
    Description:
        This function performs the agglomerative clustering with k-means. 
        
    Author:
        Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    """


    n = X.shape[1];
    k = max(r)+1;
    R = np.zeros((k,n),dtype='int');
    kmloss = np.zeros(k);
    mergeidx = np.zeros((k-1,2));
    
    mapping = np.arange(k);
    
    R[0,:] = r;

    kmloss[0] = kmeans_crit(X,r);
    
    for idx in range(1,k):
        minError = np.Inf;
        minSet1 = -1;
        minSet2 = -2;
        
        for i in range(k-idx):
            for j in range(i+1,k-idx+1):
                tempR = R[idx-1,:].copy();
                tempR[R[idx-1,:]==mapping[j]] = mapping[i];
                error = kmeans_crit(X,tempR);
                
                if error < minError:
                    minError = error;
                    minSet1 = i;
                    minSet2 = j;
        
        kmloss[idx] = minError;
        mergeidx[idx-1,:] = (mapping[minSet1], mapping[minSet2]);
        R[idx,:] = R[idx-1,:].copy();
        R[idx,R[idx-1,:]==mapping[minSet2]] = k+idx-1;
        R[idx,R[idx-1,:]==mapping[minSet1]] = k+idx-1;
        mapping[minSet1] = k+idx-1;
        mapping[minSet2] = mapping[k-idx];
        
    return R, kmloss, mergeidx;

def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    Z = np.hstack((mergeidx,kmloss[1:,np.newaxis],np.zeros((mergeidx.shape[0],1))));
    dendrogram(Z);
    d = max(Z[:,2]) - min(Z[:,2]);
    plt.ylim([max(0,min(Z[:,2])-0.01*d), max(Z[:,2])+0.01*d]);
    plt.show(block=False);
    
def log_norm_pdf(X, mu, C):
    """ Computes log probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    
    Author:
        Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    """
    if(X.ndim == 1):
        d = X.shape[0]
        n = 1
    else:
        d = X.shape[0];
        n = X.shape[1];
    centeredM = X - mu[:,np.newaxis];
    
    _,logDet = np.linalg.slogdet(C);
    iC = np.linalg.inv(C);
    
    result = np.zeros(n);
    
    for i in range(n):
        result[i] = (-d/2)*(np.log(2*np.pi))+(-1./2)*logDet+(-1./2*np.dot(centeredM[:,i],np.dot(iC,centeredM[:,i])));
    
    return result;
    
def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    
    Author:
        Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    """
    
    d = X.shape[0];
    n = X.shape[1];
    centeredM = X - mu[:,np.newaxis];
    
    determinant = np.linalg.det(C);
    iC = np.linalg.inv(C);
     
    result = np.zeros(n);
     
    # np.diag((((2*np.pi)**(-d/2))*(determinant**(-1./2))*np.exp(-1./2*(np.dot(centeredM.T,np.dot(iC,centeredM))))));
    for i in range(n):
        result[i] = (2*np.pi)**(-d/2)*determinant**(-1./2)*np.exp(-1./2*np.dot(centeredM[:,i],np.dot(iC,centeredM[:,i])))
        
    return result;

def log_likelihood(data, pi, mu, sigma):
    ''' This function calculates the log likelihood of the data given
        a mixture Gaussian distribution.
        
        Input:
            data : (d x n) data matrix with each datapoint in one column
            pi : k data array with each element specifying the probability of 
                each Gaussian component
            mu : (d x k) mean vector matrix with each mean vector in one column
            sigma : (k x d x d) covariance matrix list where the ith Gaussian component
                has the covariance matrix sigma[i].
            
        Output:
            Log likelihood of the parameters given the data points.
    '''
    k = mu.shape[1];
    n = data.shape[1];
    
    temp = np.zeros(n); 
    
    for i in range(k):
            temp += pi[i]*norm_pdf(data,mu[:,i],sigma[i]);
            
    logL = np.log(temp).sum();
    
    return logL;

def log_log_likelihood(data, pi, mu, sigma):
    ''' This function calculates the log likelihood of the data given
        a mixture Gaussian distribution.
        
        Input:
            data : (d x n) data matrix with each datapoint in one column
            pi : k data array with each element specifying the probability of 
                each Gaussian component
            mu : (d x k) mean vector matrix with each mean vector in one column
            sigma : (k x d x d) covariance matrix list where the ith Gaussian component
                has the covariance matrix sigma[i].
            
        Output:
            Log likelihood of the parameters given the data points.
    '''
    k = mu.shape[1];
    n = data.shape[1];
    
    temp = np.ones(n)*-np.Inf; 
    
    for i in range(k):
            buffer = log_norm_pdf(data,mu[:,i],sigma[i]);
            temp = log_sum(temp,np.log(pi[i])+buffer);
#             for j in range(n):
#                 temp[j] = log_sum(temp[j],np.log(pi[i]) + buffer[j]);
            
    logL = temp.sum();
    
    return logL;

def log_sum(a,b):
    assert np.ndim(a) == np.ndim(b), 'Arguments must have the same dimensions.'
    if(np.ndim(a) == 0 and np.ndim(b) == 0):
        if a == -np.Inf:
            return b
        elif b == -np.Inf:
            return a
        elif a>b:
            return a+np.log(1+np.exp(b-a));
        else:
            return b+np.log(1+np.exp(a-b));
    else:
        condlist = [a== -np.Inf, b==-np.Inf, a>b, b>=a];
        choicelist = [b, a, a+np.log(1+np.exp(b-a)), b+np.log(1+np.exp(a-b))];   
        return np.select(condlist,choicelist);

def log_em_mog(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Performs EM Mixture of Gaussians in logarithmic space

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: 1 x k matrix of priors
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    
    Author:
        Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    """
    n = X.shape[1];
    d = X.shape[0];
    pi = np.ones(k)/k;
    covariances = np.zeros((k,d,d));
    gammas = np.zeros((n,k));
    logL = -np.Inf;
    delta = 1e-5
    
    for i in range(k):
        covariances[i] = np.identity(d);
    
    if(init_kmeans == True):
        mu, _ = kmeans(X, k, max_iter);
    else:
        candidateIndices = np.random.permutation(np.arange(n))[:k];
        mu = X[:,candidateIndices];
        
    for counter in range(max_iter):
     
        sGamma = np.ones(n)*-np.Inf;
        
        for i in range(k):
            gammas[:,i] = np.log(pi[i]) + log_norm_pdf(X,mu[:,i],covariances[i]);
            sGamma = log_sum(sGamma,gammas[:,i]);
#             for j in range(n):
#                 sGamma[j] = log_sum(sGamma[j],gammas[j,i]);
            
        for i in range(k):
            gammas[:,i] = np.exp(gammas[:,i]-sGamma);
            
        nk = gammas.sum(axis=0);
        pi = nk/n;
        
            
        for i in range(k):
            mu[:,i] = np.dot(X,gammas[:,i])/nk[i];
        
            wX = (X-mu[:,i,np.newaxis]) * np.sqrt(gammas[:,i]);
            covariances[i] = np.dot(wX,wX.T)/nk[i] + np.identity(d)*delta
            
        oldLogL = logL;
            
        logL = log_log_likelihood(X, pi, mu, covariances);
        
        print('Iteration:' +str(counter) + ' log likelihood:' + str(logL));
        
        if(logL-oldLogL < eps):
            break;
        
    return pi, mu, covariances

def em_mog(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Performs EM Mixture of Gaussians

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: 1 x k matrix of priors
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    
    Author:
        Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    """
    n = X.shape[1];
    d = X.shape[0];
    pi = np.ones(k)/k;
    covariances = np.zeros((k,d,d));
    gammas = np.zeros((n,k));
    logL = -np.Inf;
    delta = 1e-5
    
    for i in range(k):
        covariances[i] = np.identity(d);
    
    if(init_kmeans == True):
        mu, _ = kmeans(X, k, max_iter);
    else:
        candidateIndices = np.random.permutation(np.arange(n))[:k];
        mu = X[:,candidateIndices];
        
    for counter in range(max_iter):
     
        sGamma = np.zeros(n);
        
        for i in range(k):
            gammas[:,i] = pi[i]*norm_pdf(X,mu[:,i],covariances[i]);
            sGamma += gammas[:,i];
            
        for i in range(k):
            gammas[:,i] /= sGamma;
            
        nk = gammas.sum(axis=0);
        pi = nk/n;
        
            
        for i in range(k):
            mu[:,i] = np.dot(X,gammas[:,i])/nk[i];
        
            wX = (X-mu[:,i,np.newaxis]) * np.sqrt(gammas[:,i]);
            covariances[i] = np.dot(wX,wX.T)/nk[i] + np.identity(d)*delta
            
        oldLogL = logL;
            
        logL = log_likelihood(X, pi, mu, covariances);
        
        print('Iteration:' +str(counter) + ' log likelihood:' + str(logL));
        
        if(logL-oldLogL < eps):
            break;
        
    return pi, mu, covariances;
        
def plot_ellipse(mean,covariance, nstd):
    values, vectors = np.linalg.eigh(covariance);
    
    ordering = np.argsort(values)[::-1];
    
    values = values[ordering[:2]];
    vectors = vectors[:,ordering[:2]];
    
    theta = np.degrees(np.arctan2(vectors[1,0],vectors[0,0]));
    
    width, height = 2*nstd*np.sqrt(values);
    
    e = Ellipse(mean, width, height, theta);
    
    ax = plt.gca();
    
    ax.add_artist(e);
    e.set_facecolor('none');
    e.set_edgecolor('black');
        

def plot_em_solution(X, mu, sigma):
    """ Plots covariance ellipses for EM MoG solution

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    
    Author:
        Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    """
    k = mu.shape[1];
    
    plt.scatter(X[0,:],X[1,:]);
    
    plt.scatter(mu[0,:],mu[1,:],c='r',marker='x');
    
    for i in range(k):
        plot_ellipse(mu[:,i],sigma[i],0.75)
        plot_ellipse(mu[:,i],sigma[i],1.5)
    
def kmean_loss(X,pi, mu, sigma):
    ''' This function calculates the km loss function given the data
        and the cluster means
        
        Input:
            X : (d x n) data matrix
            pi : 
            mu : (d x k) cluster mean matrix
            sigma :
        
        Output:
            loss function value
    '''
    n = X.shape[1];
    k = mu.shape[1];
    
    gammas = np.zeros((n,k));
    sGamma = np.ones(n)*-np.Inf;
    
    for i in range(k):
        gammas[:,i] = np.log(pi[i]) + log_norm_pdf(X,mu[:,i],sigma[i]);
        sGamma = log_sum(sGamma,gammas[:,i]);

    for i in range(k):
        gammas[:,i] = np.exp(gammas[:,i]-sGamma);
        
    randomChoice = np.random.rand(n);
    temp = np.zeros(n);
    selection = np.zeros((n,k));
    for i in range(k):
        selection[:,i] = temp+gammas[:,i] < randomChoice;
        temp += gammas[:,i];
        
    r = (selection.sum(axis=1)).astype(int);
    return kmeans_crit(X,r);
#     distanceMatrix = np.sqrt(((X[:,:,np.newaxis] - mu[:,np.newaxis,:])**2).sum(axis=0));
#     r = np.argmin(distanceMatrix,axis=1);
#     return kmeans_crit(X, r);
    
