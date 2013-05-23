import numpy as np
import matplotlib.pylab as pl

errorKM = np.array([459.38, 265.4,190.66,153.12,141.45,132.24,123.91,115.47,111.12]);
avgItKM = np.array([4.03,4.27,5.03,6.57,7.53,9.13,10.53,10.30,13.87]);
timeperItKM = np.array([4.13e-4,3.91e-4,5.96e-4,5.58e-4,6.64e-4,6.2e-4,7.91e-4,9.06e-4,9.38e-4]);

errorEM = np.array([464.3,265.42,190.67,153.11,149.84, 151.15, 145.78, 138.78,142.41]);
avgItEM = np.array([12.1, 18.43,25.77,52.93,106.47,111.5, 116.13, 192.90, 171.33]);
timeperItEM = np.array([1.29e-3,1.84e-3,2.38e-3,2.86e-3,3.36e-3,4e-3, 4.51e-3, 5.04e-3, 5.5e-3]);

errorEMKM = np.array([464.3,265.42,190.67, 153.11, 151.18, 148.62, 139.09, 142.65, 143.32]);
avgItEMKM = np.array([4.73, 4, 18.27, 46.47, 80.97, 140.73, 126.57, 155.03, 175.23]);
timeperItEMKM = np.array([1.62e-3,2.33e-3,2.72e-3, 3.34e-3, 3.92e-3, 4.57e-3, 5.22e-3, 5.87e-3, 6.5e-3]);

pl.figure();
pl.plot(np.arange(9)+2,errorKM,'o-');
pl.plot(np.arange(9)+2,errorEM,'x-');
pl.plot(np.arange(9)+2,errorEMKM,'*-');
pl.xlabel('#clusters');
pl.ylabel('Error');
pl.legend(['k-means', 'EM','EM with init']);
pl.title('Dependence of error on #clusters');
pl.show();

pl.figure();
pl.plot(np.arange(9)+2,avgItKM,'o-');
pl.plot(np.arange(9)+2,avgItEM,'x-');
pl.plot(np.arange(9)+2,avgItEMKM,'*-');
pl.xlabel('#clusters');
pl.ylabel('Avg #iterations');
pl.legend(['k-means', 'EM','EM with init']);
pl.title('Dependence of avg #iterations on #clusters');
pl.show();

pl.figure();
pl.plot(np.arange(9)+2,avgItKM*timeperItKM,'o-');
pl.plot(np.arange(9)+2,avgItEM*timeperItEM,'x-');
pl.plot(np.arange(9)+2,avgItEMKM*timeperItEMKM,'*-');
pl.xlabel('#clusters');
pl.ylabel('Convergence time in s');
pl.legend(['k-means', 'EM','EM with init']);
pl.title('Dependence of convergence time on #clusters');
pl.show();


