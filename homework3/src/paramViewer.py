#! /usr/bin/python

import pickle;
import sys;

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: paramViewer.py [file to view]';
        exit(-1);
    
    fid = open(sys.argv[1],'r');
    dics = pickle.load(fid);
    
    whitelist = set(['kernelparameter','cvloss','kernel','regularization'])
    
    for dic in dics:
        print 'Database:' + dic,;
        
        for key,value in dics[dic].items():
            if key in whitelist:
                print ' ' + str(key) +':'+ str(value),;
                
        print "";