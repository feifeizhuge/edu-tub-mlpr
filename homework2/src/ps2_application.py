import numpy as np;
import pylab as pl;
import ps2_implementation as imp;


def analyse5Gaussians():
    data = np.load('5_gaussians.npy');
    
    numClusters = np.arange(9)+2;
    max_iter = 500;
    init_kmeans = False;
    
    tries = 10;
    n = data.shape[1];
    d = data.shape[0];
    
    for num in numClusters:
        kmMinError = np.Inf;
        kmMu = np.zeros((d,num));
        kmR = np.zeros(n);
        for _ in range(tries):
            mu, r = imp.kmeans(data, num, max_iter);
            error = imp.kmeans_crit(data, r);
            
            if(error < kmMinError):
                kmMinError = error;
                kmMu = mu;
                kmR = r;
                
        pl.figure();
        pl.scatter(data[0,:],data[1,:],23,kmR);
        pl.scatter(kmMu[0,:],kmMu[1,:],23,np.arange(num),marker='x');
        error = imp.kmeans_crit(data, kmR);
        pl.title('Cluster:' + str(num) + " Error:" + str(error));
        pl.show(block=False);
        
        emLL = -np.Inf;
        emMu = np.zeros((d,num));
        emSigma = np.zeros((num,d,d));
        emPi = np.zeros(num);
        totalIterations = 0;
        
        for _ in range(tries):
            pi, mu, sigma,iterations = imp.em_mog(data,num,max_iter,init_kmeans);
            
            totalIterations += iterations
            
            ll = imp.log_likelihood(data, pi, mu, sigma);
            
            if ll > emLL:
                emLL = ll;
                emMu = mu;
                emSigma = sigma;
                emPi = pi;
        
        pl.figure();
        imp.plot_em_solution(data, emMu, emSigma);
        error = imp.kmean_loss(data,emPi, emMu, emSigma);
        pl.title('Cluster:' + str(num) + " Error:" + str(error) + " Avg iterations:" + str(float(totalIterations)/tries));
        
        pl.show(block=False);
    
    pl.figure();
    
    kmMinError = np.Inf;
    for _ in range(tries):
        _, r = imp.kmeans(data, max(numClusters), max_iter);
        error = imp.kmeans_crit(data, r);
        
        if(error < kmMinError):
            kmMinError = error;
            kmR = r;
            
    _,kmloss, mergeidx = imp.kmeans_agglo(data, kmR);
    imp.agglo_dendro(kmloss, mergeidx)
    pl.show();
    
def analyse2Gaussians():
    data = np.load('2_gaussians.npy');
    num = 2;
    d = data.shape[0];
    n = data.shape[1];
    max_iter =500;
    init_kmeans = True;
    tries = 10;
    minKMError = np.Inf;
    maxEMLL = -np.Inf;
    minMu = np.zeros((d,num));
    minR = np.zeros(n);
    
    for _ in range(tries):
        mu, r = imp.kmeans(data,num,max_iter);
        
        error = imp.kmeans_crit(data, r);
        
        if minKMError > error:
            minKMError = error;
            minMu = mu;
            minR = r;
    
    print("Min error:" + str(minKMError));
    pl.figure();
    pl.scatter(data[0,:],data[1,:],23,minR);
    pl.scatter(minMu[0,:],minMu[1,:],23,np.arange(num),marker='x');
    pl.title('Kmean error:' + str(minKMError));
    maxMu = np.zeros((d,num));
    maxSigma = np.zeros((num,d,d));
    maxPi = np.zeros(num);
    
    for _ in range(tries):    
        pi, mu, sigma = imp.log_em_mog(data,num,max_iter,init_kmeans);
        
        ll = imp.log_likelihood(data, pi, mu, sigma);
        
        if maxEMLL < ll:
            maxEMLL = ll;
            maxMu = mu;
            maxSigma = sigma;
            maxPi = pi;
        
    print("Max log likelihood:" + str(maxEMLL));
    
    pl.figure();
    
    imp.plot_em_solution(data, maxMu, maxSigma);
    pl.title('EM error:' + str(imp.kmean_loss(data, maxPi, maxMu, maxSigma)))
    pl.show();
    
def analyseUSPS():
    datafile = np.load('usps.npz');
    
    data = datafile['data_patterns'];
    
    #n = data.shape[1];
    d = data.shape[0];
    kmMax_iter = 500;
    emMax_iter = 100;
    k = 10;
    tries = 10;
    minKMError = np.Inf;
    maxEMLL = -np.Inf;
    init_kmeans = False;
    
    kmMu = np.zeros((d,k));
    #kmR = np.zeros(n);
    emMu = np.zeros((d,k));
    emSigma = np.zeros((k,d,d));
    emPi = np.zeros(k);
     
    for _ in range(tries):
        mu,r = imp.kmeans(data, k, kmMax_iter);
          
        error = imp.kmeans_crit(data, r);
          
        if error < minKMError:
            minKMError = error;
            kmMu = mu;
            #kmR = r;
            
    for _ in range(tries):
        pi,mu,sigma = imp.log_em_mog(data, k, emMax_iter, init_kmeans);
         
        ll = imp.log_log_likelihood(data, pi, mu, sigma);
         
        if ll > maxEMLL:
            maxEMLL = ll;
            emMu = mu;
            emPi = pi
            emSigma = sigma;
             
    print('KM error:' + str(minKMError) + ' EM error:' +str(imp.kmean_loss(data,emPi,emMu,emSigma)));
     
    pl.figure();
    for i in range(10):
        pl.subplot(3,4,i);
        pl.imshow(kmMu[:,i].reshape(16,16));
        pl.axis('off');
        
    pl.suptitle('KMeans');
    
    pl.figure();
    for i in range(10):
        pl.subplot(3,4,i);
        pl.imshow(emMu[:,i].reshape(16,16));
        pl.axis('off');
        
    pl.suptitle('EM');
        
    pl.show();
    
def dendrogramUSPS():
    datafile = np.load('usps.npz');
    
    data = datafile['data_patterns'];
       
    n = data.shape[1]; 
    k = 10;
    max_iter = 100;
    tries = 10;
    kmMinError = np.Inf;
    kmR = np.zeros(n);
    
    for i in range(tries):
        _,r = imp.kmeans(data, k, max_iter);
        
        error = imp.kmeans_crit(data, r);
        
        if error < kmMinError:
            kmMinError = error;
            kmR = r;
    
    R, kmloss, mergeidx = imp.kmeans_agglo(data, kmR);
    imp.agglo_dendro(kmloss, mergeidx);
    
    m = np.ceil(np.sqrt(k));
    n = np.ceil(float(k)/m);
    maxCluster = max(R[0,:])+1;
    counter = 0;
    pl.figure();
    for j in range(maxCluster):
        if (R[0,:] == j).sum() > 0:
            mu = data[:,R[0,:]==j].sum(axis=1)/(R[0,:]==j).sum();
            pl.subplot(m,n,counter);
            pl.imshow(mu.reshape(16,16));
            counter += 1;
            pl.axis('off');
    pl.suptitle('All centroids');
    
    
    for i in range(1,k):
        leftMu = data[:,R[i-1,:]==mergeidx[i-1,0]].sum(axis=1)/(R[i-1,:]==mergeidx[i-1,0]).sum();
        rightMu = data[:,R[i-1,:] == mergeidx[i-1,1]].sum(axis=1)/(R[i-1,:]==mergeidx[i-1,1]).sum();
        newMu = (data[:,R[i-1,:]==mergeidx[i-1,0]].sum(axis=1) + data[:,R[i-1,:]==mergeidx[i-1,1]].sum(axis=1))/((R[i-1,:]==mergeidx[i-1,0]).sum()+ (R[i-1,:]==mergeidx[i-1,1]).sum()) ;
        pl.figure();
        pl.subplot(1,3,1);
        pl.imshow(leftMu.reshape(16,16));
        pl.axis('off');
        pl.title('Left cluster centroid');
        pl.subplot(1,3,2);
        pl.imshow(rightMu.reshape(16,16));
        pl.axis('off');
        pl.title('Right cluster centroid');
        pl.subplot(1,3,3);
        pl.imshow(newMu.reshape(16,16));
        pl.axis('off');
        pl.title('Merged cluster centroid');
        pl.suptitle('Merge:'+str(i));
        
    pl.show();

if __name__ == '__main__':
#     analyse5Gaussians();
#     analyse2Gaussians();
#     analyseUSPS();
    dendrogramUSPS();