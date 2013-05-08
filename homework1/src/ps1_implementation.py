""" sheet1_implementation.py

PUT YOUR NAME HERE:
Till Rohrmann, Matrikelnummer: 343756


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la, scipy.sparse as sp
import pylab as pl
import copy


def pca(X, m):
    ''' pca - computes the principal components for the given data
    
        usage: 
            Z, U, D = pca(X,m)
        
        input:
            X : (d,n)-array of column vectors indicating the observations
            m : number of principal components used for the projection
            
        output:
            Z : (m,n)-array of column vectors containing the projected data
            U : (d,d)-array of column vectors being the principal components
            D : (1,d)-array of principal values in decreasing order
            
        description:
            This function calculates for the given input data the principal components 
            and principal values. Furthermore it projects the given data on the first
            m principal components and returns this projection.
        
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    mean = np.mean(X, axis=1);
    X -= mean[:, np.newaxis];
    
    CovarianceMatrix = np.dot(X, X.T) / (X.shape[1] - 1);
    
    w, U = la.eigh(CovarianceMatrix);
    D = w[::-1];
    U = U[:, ::-1];
        
    Z = np.dot(U[:, 0:m].T, X);
    
    return (Z, U, D)
    
def gammaidx(X, k):
    ''' gammaidx - computes the gamma index for every point
    
        usage:
            y = gammaidx(X, k)
            
        input:
            X : (d,n)-array of column vectors indicating the observations
            k : number of neighbors to be taken into consideration
            
        output:
            y : (1,n)-array containing for every observation the gamma index
            
        description:
            This function calculates for every observation its gamma index wrt the number
            of neighbors.
        
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    n = X.shape[1];
    distanceMatrix = calcEuclideanDistance(X);
    
    
    # using the build-in method seems to be faster than the quickFindFirstK algorithm
    indices = np.argsort(distanceMatrix);
    indices = indices[:, 0:k + 1];
    rows = np.arange(n)[:, np.newaxis];
    gamma = distanceMatrix[rows, indices].sum(axis=1) / k;
    gamma = gamma[np.newaxis, :];
        
    return gamma;
    

def calcEuclideanDistance(X):
    ''' calcEuclideanDistance - computes the pairwise euclidean distance between
        the observations contained in X
        
        usage:
            D = calcEuclideanDistance(X)
            
        input:
            X : (d,n)-array of column vectors representing the observations
            
        output:
            D : (n,n)-array containing the pairwise euclidean distances
            
        description:
            This functions calculates for each pair of observations i,j the euclidean
            distance and stores it at the index (i,j) in the resulting distance matrix.
            
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    rows = X.T[:, np.newaxis, :];
    columns = X.T[np.newaxis, :, :];
    
    diff = rows - columns;
    
    sqDiff = diff ** 2;
    
    D = np.sqrt(sqDiff.sum(axis=2))
    
    return D;

def quickFindFirstK(data, k):
    ''' quickFindFirstK - finds the smallest k elements in data
    
        usage:
            elems, order = quickFindFirstK(data, k)
            
        input:
            data : (n)-array of values
            k : number of smallest components to extract from data
        
        output:
            elems : (k)-array containing the k smallest elements
            order : list containing the positions in the original data from where the
                k smallest elements stem
                
        description:
            This function finds the k smallest elements in data and returns them.
            
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    n = data.shape[0];
    
    order = np.arange(n);
    copy = data.copy();
    
    quickFindFirstKHelper(copy, 0, n - 1, k, order);
    
    return (copy[0:k], order[0:k].tolist());

def quickFindFirstKHelper(data, left, right, k, order):
    ''' quickFindFirstKHelper - Helper function for quickFindFirstK performing the actual selection of
            the k smallest elements from data
            
        usage:
            quickFindFirstKHelper(data, left, right, k, order)
            
        input:
            data : data array
            left : left index for selection
            right : right index for selection
            k : number of elements to select
            order : array containing the original position of the sorted elements
        
        output:
            none
            
        description:
            This function places the k smallest elements at the front of the data array
            
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    if right > left:
        pivotValue, pivotIndex = selectPivot(data, left, right);
        
        temp = data[pivotIndex];
        data[pivotIndex] = data[right];
        data[right] = temp;
        
        temp = order[pivotIndex]
        order[pivotIndex] = order[right];
        order[right] = temp;
        
        index = left;
        
        for i in range(left, right):
            if(data[i] <= pivotValue):
                temp = data[i];
                data[i] = data[index];
                data[index] = temp;
                temp = order[i];
                order[i] = order[index];
                order[index] = temp;
                index += 1;
            
        data[right] = data[index];
        data[index] = pivotValue;
        temp = order[right]
        order[right] = order[index];
        order[index] = temp;
        
        if index < left + k - 1:
            quickFindFirstKHelper(data, index + 1, right, k - index + left - 1, order);
        elif index > left + k - 1:
            quickFindFirstKHelper(data, left, index - 1, k, order);
    
def selectPivot(data, left, right):
    ''' selectPivot - selects a pivot element from data
    
        usage:
            pivotValue, pivotIndex = selectPivot(data, left, right)
            
        input:
            data : (n)-array containing the data
            left : left index boundary
            right: right index boundary
            
        output:
            pivotValue : pivotValue element
            pivotIndex : index of pivot element
            
        description:
            This function selects a pivot element from the data and returns
            its value and index.
            
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    pivotIndex = np.random.randint(left, right + 1);
    pivotValue = data[pivotIndex];
    
    return (pivotValue, pivotIndex);

def plotNeighborhoodGraph(X, neighbors, dimensions=(0, 1)):
    ''' plotNeighborhoodGraph - plots the neighborhood graph for the data points X and 
        the given neighborhood relationship
        
        usage:
            plotNeighborhoodGraph(X,neighbors)
            
        input:
            X : (d,n)-array containing the observations as column vectors
            neighbors : list containing as lists the neighbors for each observation
            dimensions : tuple of 2 dimensions which shall be plotted
            
        description:
            This function plots the given data points and its neighborhood relationships
            
        author:
            Till Rohrmann - till.rohrmann@campus.tu-berlin.de
    '''
    
    pl.figure();
    pl.hold(True);
    pl.scatter(X[dimensions[0], :], X[dimensions[1], :]);
    pl.title('Neighborhood Graph');
    pl.xlabel('x' + str(dimensions[0]));
    pl.ylabel('x' + str(dimensions[1]));
    
    for index, nlist in enumerate(neighbors):
        x = X[:, index];
         
        for neighbor in nlist:
            xNeighbor = X[:, neighbor];
            pl.plot([x[0], xNeighbor[0]], [x[1], xNeighbor[1]], 'b-');
                    
    pl.hold(False);

    pl.show(block=False);
    
def checkWeaklyConnectedGraph(neighbors):
    ''' checkWeaklyConnectedGraph - checks whether the directed neighborhood
        graph is weakly connected.
        
        usage:
            b = checkStronglyConnectedGraph(X,neighbors)
            
        input:
            neighbors : list of list of neighbors for each node in the graph
        
        output:
            b : boolean saying whether the graph is connected
            
        description:
            This function checks whether the given graph is weakly connected or not.
        
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    n = len(neighbors);
    
    neighbors = copy.deepcopy(neighbors);
    
    # establish unidirectional edges
    for index, neighborhood in enumerate(neighbors):
        for neighbor in neighborhood:
            if neighbor > index and not index in neighbors[neighbor]:
                neighbors[neighbor].append(index);
    
    stack = [0];
    visited = set();
    
    while len(stack) > 0:
        elem = stack.pop();
        visited.add(elem);
        
        for neighbor in neighbors[elem]:
            if not neighbor in visited:
                stack.append(neighbor);
    
    return len(visited) == n;
    
    
def checkStronglyConnectedGraph(neighbors):
    ''' checkStronglyConnectedGraph - checks whether the neighborhood graph is strongly
        connected by using Tarjan's algorithm
    
        usage:
            b = checkStronglyConnectedGraph(X,neighbors)
            
        input:
            neighbors : list of list of neighbors for each node in the graph
        
        output:
            b : boolean saying whether the graph is connected
            
        description:
            This function checks whether the given graph is connected or not.
            This is done by using Tarjan's algorithm.
        
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    n = len(neighbors);
    index = 0;
    nodeIndices = np.ones(n) * -1;
    nodeLowlink = np.ones(n) * -1;
    
    strongconnect(0, neighbors, index, nodeIndices, nodeLowlink, list());
    
    return all(elem == 0 for elem in nodeLowlink);

def strongconnect(node, neighbors, index, nodeIndices, nodeLowlink, stack):
    ''' strongconnect - calculates the strongly connected component with node
        being its root
        
        usage:
            strongconnect(node, neighbors, index, nodeIndices, nodeLowlink)
        
        input:
            node : integer denoting the starting node
            neighbors : list of int lists containing for each node its neighborhood 
                relationship
            index : current counting index
            nodeIndices : list storing the currently assigned indices of the nodes
            nodeLowlink : list storing the currently lowest index reachable from the 
                current node
            stack : containing the current set of nodes constituting the current strongly
                connected component
                
        description:
            This function is part of the Tarjan's algorithm for the detection of 
            strongly connected components in an directed graph. See
            http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
            for further explanation.
            
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    nodeIndices[node] = index;
    nodeLowlink[node] = index;
    stack.append(node);
    index += 1;
    
    for neighbor in neighbors[node]:
        if(nodeIndices[neighbor] == -1):
            strongconnect(neighbor, neighbors, index, nodeIndices, nodeLowlink);
            nodeLowlink[node] = min(nodeLowlink[node], nodeLowlink[neighbor]);
        elif neighbor in stack:
            nodeLowlink[node] = min(nodeLowlink[node], nodeIndices[neighbor]);
            
    if nodeIndices[node] == nodeLowlink[node]:
        n = stack.pop();
        
        while(n != node):
            n = stack.pop();

def lle(X, m, n_rule, param, tol=1e-2):
    ''' lle - computes the locally linear embedding of the data points in X
    
        usage:
            Y = lle(X, m, n_rule, param, tol);
            
        input:
            X : (d,n)-array containing the observations as column vectors
            m : dimension of embedded points Y
            n_rule : neighbor-selection strategy: 
                'knn' : k nearest neighbors
                'eps-ball' : all points lying in the ball around the data point
            param : Parameter for the neighbor-selection strategy
            tol : regularization parameter
            
        output:
            Y : (m,n)-array containing the coordinates of the embedded observations
            
        description:
            This function calculates a m-dimensional embedding of the given data points
            in X.
            
        author:
            Till Rohrmann, till.rohrmann@campus.tu-berlin.de
    '''
    
    assert tol > 0, 'Regularization parameter has to be positive';
    
    print 'Step 1: Finding the nearest neighbors by rule ' + n_rule
    
    distanceMatrix = calcEuclideanDistance(X);
    neighbors = list();
    n = X.shape[1];
    
    distanceMatrix[np.arange(n), np.arange(n)] = np.Inf;
    
    if n_rule == 'knn':
        index = np.argsort(distanceMatrix)[:, 0:param];
        for row in index:
            neighbors.append(row.tolist());
    elif n_rule == 'eps-ball':
        selectedElements = distanceMatrix <= param;
        
        for index, row in enumerate(selectedElements):
            s = row.nonzero()[0].tolist();
            
            if len(s) == 0:
                raise Exception('Point ' + str(index) + ' has no neighbors')
            
            neighbors.append(s);
    else:
        raise Exception('Invalid argument value n_rule=' + n_rule);
    
    print 'Step 2: local reconstruction weights'
    
    if(not checkWeaklyConnectedGraph(neighbors)):
        raise Exception('Graph is not weakly connected');
    
    W = sp.lil_matrix((n, n))
    
    for index, observation in enumerate(X.T):
        tempData = X[:, neighbors[index]].copy();
        tempData = observation[:, np.newaxis] - tempData;
        numNeighbors = len(neighbors[index]);
        C = np.dot(tempData.T, tempData) + np.identity(numNeighbors) * tol / numNeighbors;
        ones = np.ones((numNeighbors, 1));
        w = la.solve(C, ones);
        w /= sum(w);
        W[index, neighbors[index]] = w;
        
    W = W.tocsr();
    
    print 'Step 3: compute embedding'
    
    Id = sp.identity(n);
    
    # For better performance one could exploit the block structure of W
    M = ((Id - W).T * (Id - W)).todense();
    
    _, eigenvectors = la.eigh(M, eigvals=(0, m));
    
    return eigenvectors[:, 1:m + 1].T;
