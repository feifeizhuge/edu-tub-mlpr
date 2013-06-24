import pickle;
import numpy as np;
import matplotlib.pylab as pl
import matplotlib.cm as cm

import ps3_implementation as imp

if __name__ == "__main__":
    filename = "results/results.p"
    
    fid = open(filename,'rb');
    file = pickle.load(fid);
    
    xTest = np.loadtxt("ps3_datasets/U04_banana-xtest.dat");
    yTest = file['banana']['ypred']
    
    xTrain = np.loadtxt("ps3_datasets/U04_banana-xtrain.dat");
    yTrain = np.loadtxt("ps3_datasets/U04_banana-ytrain.dat");
    
    n = xTest.shape[1];
    
    lables = np.ones((1,n));
    
    lables[yTest >=0] = 0.5;
# 
#     pl.scatter(xTest[0,:],xTest[1,:],c=lables, cmap = cm.jet);
#     pl.title('Prediction banana test data set');
#     
#     pl.figure();
#     pl.scatter(xTrain[0,:],xTrain[1,:],c=yTrain, cmap=cm.jet);
#     pl.title('Banana training data set');
    
    krr = imp.krr(file['banana']['kernel'],file['banana']['kernelparameter'],file['banana']['regularization']);
    
    krr.fit(xTrain,yTrain);
    xInput = np.linspace(-3, 3);
    (x,y) = np.meshgrid(xInput, xInput);
    
    x = x.reshape((1,x.size));
    y = y.reshape((1,y.size));
    X = np.vstack((x,y));
    krr.predict(X);
    Z = krr.ypred;
    sn = np.sqrt(x.size);
    Z = Z.reshape((sn,sn));
    
    pl.figure();
     
    pl.contour(xInput,xInput,Z,5);
    pl.colorbar();
    pl.title('Contour plot of banana\'s classification function');
    
    pl.show()