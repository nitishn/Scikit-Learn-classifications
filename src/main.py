'''
Created on May 6, 2012

@author: nitish
'''
import numpy as np
import pylab as pl
from sklearn import datasets, svm, neighbors
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from datetime import datetime

def neigh():
##########################################################################
############## Dataset A                     #############################
########################################################################## 
    i = datasets.make_classification(n_samples=400, n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    j = datasets.make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=4, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    k = datasets.make_classification(n_samples=100, n_features=200, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)   
    
    n_neighbors = 2
    
    X = i[0]
    y = i[1]    
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure()
        pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        pl.title(" Nearest Neighbor 2D, 2-Class classification (k = %i, weights = '%s')"
                 % (n_neighbors, weights))
        pl.axis('tight')

##########################################################################
############## Dataset B                     #############################
##########################################################################    
    X = j[0]
    y = j[1]    
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FF00FF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', "#DC143C"])

    for weights in ['uniform']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure()
        pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        pl.title("Nearest Neighbor 2D, 4-Class classification (k = %i, weights = '%s')"
                 % (n_neighbors, weights))
        pl.axis('tight')
        
##########################################################################
############## Dataset C                     #############################
##########################################################################
    X = k[0]
    y = k[1]    
    h = .02  # step size in the mesh

    for weights in ['uniform']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)        

        
def reg():     
    i = datasets.make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    j = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=4, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    k = datasets.make_classification(n_samples=100, n_features=200, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    
##########################################################################
############## Dataset A                     #############################
##########################################################################

    X = i[0]
    y = i[1]
    
    X = X[y != 2]
    y = y[y != 2]

    X -= np.mean(X, 0)
    
    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)


    print "Computing regularization path 2D, 2 classes..."
    start = datetime.now()
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
            clf.set_params(C=c)
            clf.fit(X, y)
            coefs_.append(clf.coef_.ravel().copy())
    print "This took ", datetime.now() - start

    pl.figure()
    coefs_ = np.array(coefs_)
    pl.plot(np.log10(cs), coefs_)
    ymin, ymax = pl.ylim()
    pl.xlabel('log(C)')
    pl.ylabel('Coefficients')
    pl.title('Logistic Regression Path 2D, 2-Classes')
    pl.axis('tight')
    
    
##########################################################################
############## Dataset B                     #############################
##########################################################################
    X = j[0]
    y = j[1]
    
    X = X[y != 2]
    y = y[y != 2]

    X -= np.mean(X, 0)
    
    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)

    print "Computing regularization path 2D, 4 classes..."
    start = datetime.now()
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
            clf.set_params(C=c)
            clf.fit(X, y)
            coefs_.append(clf.coef_.ravel().copy())
    print "This took ", datetime.now() - start

    pl.figure()
    coefs_ = np.array(coefs_)
    pl.plot(np.log10(cs), coefs_)
    ymin, ymax = pl.ylim()
    pl.xlabel('log(C)')
    pl.ylabel('Coefficients')
    pl.title('Logistic Regression Path 2D, 4 classes')
    pl.axis('tight')
       
##########################################################################
############## Dataset C                     #############################
##########################################################################    
    ''''X = k[0]
    y = k[1]
    
    X = X[y != 2]
    y = y[y != 2]

    X -= np.mean(X, 0)

    ###############################################################################
    # Demo path functions

    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)


    print "Computing regularization path 200D, 2-Classes..."
    start = datetime.now()
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
            clf.set_params(C=c)
            clf.fit(X, y)
            coefs_.append(clf.coef_.ravel().copy())
    print "This took ", datetime.now() - start

    pl.figure()
    coefs_ = np.array(coefs_)
    pl.plot(np.log10(cs), coefs_)
    ymin, ymax = pl.ylim()
    pl.xlabel('log(C)')
    pl.ylabel('Coefficients')
    pl.title('Logistic Regression Path 200D, 2-Classes')
    pl.axis('tight')'''
       
       
def vec():
    i = datasets.make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    j = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=4, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    k = datasets.make_classification(n_samples=100, n_features=200, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    
##########################################################################
############## Dataset A                     #############################
##########################################################################    
    X = i[0]  # we only take the first two features.                         
    Y = i[1]
    
    h = .02  # step size in the mesh
    
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = svm.LinearSVC(C=C).fit(X, Y)
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # title for the plots
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel',
              'LinearSVC (linear kernel)']
    
    pl.figure()
    for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
        # Plot the decision boundary. For that, we will asign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        pl.subplot(2, 2, i + 1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
        pl.axis('off')
    
        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    
        pl.title(titles[i] + "2D, 2-Classes")
        
##########################################################################
############## Dataset B                     #############################
##########################################################################      
    X = j[0]  # we only take the first two features.                         
    Y = j[1]
    
    h = .02  # step size in the mesh
    
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = svm.LinearSVC(C=C).fit(X, Y)
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # title for the plots
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel',
              'LinearSVC (linear kernel)']
    
    pl.figure()
    for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
        # Plot the decision boundary. For that, we will asign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        pl.subplot(2, 2, i + 1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
        pl.axis('off')
    
        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    
        pl.title(titles[i] + "2D, 4-Classes")
##########################################################################
############## Dataset C                     #############################
##########################################################################   
    ''''X = k[0]                           
    Y = k[1]
    
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = svm.LinearSVC(C=C).fit(X, Y)

    for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])'''
    

   
def printGraphs():
    pl.show()
    
def main():
    neigh()
    reg()
    vec()
    printGraphs()

if __name__ == '__main__':
    main()
