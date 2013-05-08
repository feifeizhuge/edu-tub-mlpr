""" sheet1_implementation.py

PUT YOUR NAME HERE:
Till Rohrmann, Matrikelnummer: 343756


Write the functions
- usps
- outliers
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

import ps1_implementation as imp
imp = reload(imp)


def usps():
    ''' performs the usps analysis for assignment 4'''
    
    datafile = np.load('usps.npz');
    digitData = datafile['data_patterns'];
    digitLabels = datafile['data_labels'];

    digits = (digitLabels == 1).T.nonzero()[1];
    wnStdDeviation = 1.5;
    noisyData = digitData + wnStdDeviation*np.random.randn(digitData.shape[0],digitData.shape[1]);
    denoisedData = np.zeros((digitData.shape[0],digitData.shape[1]));
    
    for digit in range(0,10):
        data = digitData[:,digits==digit];
        datamean = np.mean(data,axis=1);
        _,pcs,pvs = imp.pca(data,0);
         
        ndata = noisyData[:,digits==digit];
        ndatamean = np.mean(ndata,axis=1);
        _,_,npvs = imp.pca(ndata,0);
           
        pl.figure();
        pl.subplot(221);
        pl.title('All principal values');
        pl.bar(range(pvs.shape[0]), pvs);
        pl.xlim((0,pvs.shape[0]));
        pl.ylim((0,pvs[0]*1.1));
           
        pl.subplot(222);
        pl.title('25 biggest principal values');
        pl.bar(range(min(25,pvs.shape[0])),pvs[0:min(25,pvs.shape[0])]);
        pl.xlim((0,min(25,pvs.shape[0])));
        pl.ylim((0,pvs[0]*1.1));
          
        pl.subplot(223);
        pl.title('Noisy data: All principal values');
        pl.bar(range(npvs.shape[0]), npvs);
        pl.xlim((0,npvs.shape[0]));
        pl.ylim((0,npvs[0]*1.1));
           
        pl.subplot(224);
        pl.title('Noisy data: 25 biggest principal values');
        pl.bar(range(min(25,npvs.shape[0])),npvs[0:min(25,npvs.shape[0])]);
        pl.xlim((0,min(25,npvs.shape[0])));
        pl.ylim((0,npvs[0]*1.1));
           
        pl.suptitle('Digit:' + str(digit));
           
        pl.figure();
        pl.subplot(2,3,1);
        pl.title('Mean');
        pl.imshow(datamean.reshape((16,16)));
           
        for i in range(5):
            pl.subplot(2,3,2+i);
            pl.title('PC'+str(i));
            pl.imshow(pcs[:,i].reshape((16,16)));
           
        pl.suptitle('Digit:' + str(digit));
    
    
    pl.figure();
    examples = [0,16,15,7,8];
    
    for i,index in enumerate(examples):
        ndata = noisyData[:,digits == digits[index]];
        ndatamean = np.mean(ndata,axis=1);
    
        _,npcs,npvs = imp.pca(ndata,0);
        minError = np.Inf;
        minErrorM = -1;
        for m in range(0,10,1):
            basis = npcs[:,0:m+1];
            denoisedData[:,index] = np.dot(basis,np.dot(basis.T,noisyData[:,index]-ndatamean)) + ndatamean;
            error = sum((digitData[:,index] - denoisedData[:,index])**2);
            
            if error < minError:
                minError = error;
                minErrorM = m;
        
        basis = npcs[:,0:minErrorM+1];
        denoisedData[:,index] = np.dot(basis,np.dot(basis.T,noisyData[:,index]-ndatamean)) + ndatamean;

        pl.subplot(5,3,1+3*i);
        pl.imshow(digitData[:,index].reshape((16,16)));
        pl.title('Original')
        pl.xticks([]);
        pl.yticks([]);
        pl.subplot(5,3,2+3*i);
        pl.imshow(noisyData[:,index].reshape((16,16)));
        pl.title('Noisy');
        pl.xticks([]);
        pl.yticks([]);
        pl.subplot(5,3,3+3*i);
        pl.imshow(denoisedData[:,index].reshape((16,16)));
        pl.xticks([]);
        pl.yticks([]);
        pl.rcParams.update({'font.size': 8})
        pl.title('M:' + str(minErrorM+1) + ' Error:' + str(error))
    
    pl.suptitle('Denoising');  
    pl.show();
    
        
def outliers_calc():
    ''' outlier analysis for assignment 5'''
    datafile = np.load('banana_data.npz');
    labels = datafile['label'];
    data = datafile['data'];
    positives = data[:,(labels==1).flatten()];
    negatives = data[:,(labels==-1).flatten()];
    ratios = [0.01,0.05,0.1,0.25];
    ks = [3,5];
    measurements =100;
    numPositives = positives.shape[1];
    numNegatives = negatives.shape[1];
    aucs = np.zeros((4,3,measurements));
    numMeasurePoints = 100;
    
    for rindex,ratio in enumerate(ratios):
        num = min(numNegatives, int(numPositives*ratio));
        
        for i in range(measurements):
            print i
            outliers = selectRandomly(negatives.T,num).T;
            dataset = np.concatenate((positives, outliers),axis=1);
            
            for index,k in enumerate(ks):
                gamma = imp.gammaidx(dataset, k);
                
                measurePoints = np.linspace(0,np.max(gamma),num= numMeasurePoints);
                
                selection = gamma <= measurePoints[:,np.newaxis];
                
                positiveClassSelection = selection[:,0:numPositives];
                negativeClassSelection = selection[:,numPositives:];
                
                tpr = sum(positiveClassSelection.T)/float(numPositives);
                fpr = sum(negativeClassSelection.T)/float(num);
                
                assert len(measurePoints) > 1, 'There has to be at least 2 measure points.'
                
                y = (tpr[:-1] + tpr[1:])/2;
                x = fpr[1:] - fpr[:-1];
                
                # use simple trapezoidal rule for integration
                aucs[rindex,index,i] = np.dot(y.T,x);
            
            meanData = np.mean(dataset,axis=1);
            distance2Mean = sum(dataset - meanData[:,np.newaxis]**2);
            
            measurePoints = np.linspace(0,max(distance2Mean),num = numMeasurePoints);
            
            selection = distance2Mean[np.newaxis,:] < measurePoints[:,np.newaxis];
            
            positiveClassSelection = selection[:,0:numPositives];
            negativeClassSelection = selection[:, numPositives:];
            
            tpr = sum(positiveClassSelection.T)/float(numPositives);
            fpr = sum(negativeClassSelection.T)/float(num);
            
            y = (tpr[:-1]+tpr[1:])/2;
            x = fpr[1:]-fpr[:-1];
            
            aucs[rindex,2,i] = np.dot(y.T,x);
            
    np.savez_compressed('outliers', ratios=ratios, aucs = aucs);
    
def selectRandomly(X,num):
    ''' selectRandomly - selects num randomly chosen points from X
    
        usage:
            R = selectRandomly(X,num)
            
        input:
            X : (d,n)-array of data points as column vectors
            num : number of randomly chosen points
            
        output:
            R : (d,num)-array containing the randomly chosen points
            
        description:
            This function selects randomly num distinct points from X
            and returns them. It selects the points from the first axis.
            
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    return np.random.permutation(X)[0:num];
      
def outliers_disp():
    ''' display the boxplots'''
    results = np.load('outliers.npz')
    aucs = results['aucs'];
    ratios = results['ratios'];
    
    pl.figure();
    m = np.ceil(np.sqrt(len(ratios)));
    n = np.ceil(len(ratios)/m)
    for i,ratio in enumerate(ratios):
        pl.subplot(m,n,i+1);
        pl.boxplot(aucs[i].T,vert=0);
        pl.title('Contamination rate:' + str(ratio) +'%');
        pl.xlim((0,1));
        pl.yticks([1,2,3],['k=3','k=5','mean']);
    
    pl.show();          

def lle_visualize(dataset='flatroll'):
    ''' visualization of LLE for assignment 6'''
    
    if dataset == 'flatroll':
        datafile = np.load('flatroll_data.npz');
        data = datafile['Xflat'];
        trueEmbedding = datafile['true_embedding'];
        #15, 1e-1
        Y = imp.lle(data, 1, 'knn', 9, 18e-3);
        
        pl.figure();
        pl.subplot(1,2,1);
        pl.scatter(data[0,:],data[1,:],c=trueEmbedding);
        pl.title('True embedding');
        pl.colorbar();
        pl.subplot(1,2,2);
        pl.scatter(data[0,:],data[1,:],c=Y);
        pl.title('LLE embedding');
        pl.colorbar();
        
        pl.show();
    
    if dataset == 'swissroll':
        datafile = np.load('swissroll_data.npz');
        
        data = datafile['x_noisefree'];
        trueEmbedding = datafile['z'];
        
        #6,1e-4
        Y = imp.lle(data,2,'knn',6,1e-4);
    
        fig = pl.figure();
        ax = fig.add_subplot(221, projection='3d');
        ax.scatter(data[0,:],data[1,:],data[2,:],c=trueEmbedding[0,:]);
        pl.title('True embedding');
        ax = fig.add_subplot(222, projection='3d');
        ax.scatter(data[0,:],data[1,:],data[2,:],c=Y[0,:]);
        pl.title('LLE embedding');
        
        pl.subplot(2,2,3);
        pl.scatter(trueEmbedding[0,:],trueEmbedding[1,:],c=trueEmbedding[0,:]);
        pl.title('True embedding');
        
        pl.subplot(2,2,4);
        pl.scatter(Y[0,:],Y[1,:],c=Y[0,:]);
        pl.title('LLE embedding');
        pl.show();
        
    if dataset == 'fishbowl':
        datafile = np.load('fishbowl_data.npz');
        
        data = datafile['x_noisefree'];
        trueEmbedding = datafile['z'];
        
        Y = imp.lle(data,2,'knn',7,1e-10);
    
        fig = pl.figure();
        ax = fig.add_subplot(221, projection='3d');
        ax.scatter(data[0,:],data[1,:],data[2,:],c=trueEmbedding[0,:]);
        pl.title('True embedding');
        ax = fig.add_subplot(222, projection='3d');
        ax.scatter(data[0,:],data[1,:],data[2,:],c=Y[0,:]);
        pl.title('LLE embedding');
        
        pl.subplot(2,2,3);
        pl.scatter(trueEmbedding[0,:],trueEmbedding[1,:],c=trueEmbedding[0,:]);
        pl.title('True embedding');
        
        pl.subplot(2,2,4);
        pl.scatter(Y[0,:],Y[1,:],c=Y[0,:]);
        pl.title('LLE embedding');
        pl.show();

def lle_noise():
    ''' LLE under noise for assignment 7'''
    datafile = np.load('flatroll_data.npz');
    data = datafile['Xflat'];
    
    noisyData1 = data + np.random.randn(data.shape[0],data.shape[1])*np.sqrt(0.2);
    noisyData2 = data + np.random.randn(data.shape[0],data.shape[1])*np.sqrt(1.8);  
    
    Y1 = imp.lle(noisyData1, 1, 'knn', 14, 1e0);
    Y2 = imp.lle(noisyData1, 1, 'knn',50,1e-2);
    
    Y3 = imp.lle(noisyData2, 1, 'knn',7,1e-1);
    Y4 = imp.lle(noisyData2, 1, 'knn', 50, 1e-2);
    
    pl.figure();
    pl.subplot(2,2,1);
    pl.scatter(noisyData1[0,:],noisyData1[1,:],c=Y1);
    pl.title('Low noise, k=14');
    
    pl.subplot(2,2,2);
    pl.scatter(noisyData1[0,:],noisyData1[1,:],c=Y2);
    pl.title('Low noise, k=50');
    
    pl.subplot(2,2,3);
    pl.scatter(noisyData2[0,:],noisyData2[1,:],c=Y3);
    pl.title('High noise, k=7');
    
    pl.subplot(2,2,4);
    pl.scatter(noisyData2[0,:],noisyData2[1,:],c=Y4);
    pl.title('High noise, k=50');
    
    pl.show();
    
    
if __name__ == "__main__":
    usps();
    outliers_calc();
    outliers_disp();
    lle_visualize('flatroll');
    lle_visualize('swissroll');
    lle_visualize('fishbowl');
    lle_noise();


