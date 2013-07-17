from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
import sklearn.ensemble #.RandomForestClassifier
from scipy import sparse
from itertools import combinations
from scipy.optimize import fmin

import numpy as np
import pandas as pd

import collections
import sys, traceback

def log_exception(*s):
    with open('resources/exceptions.txt','a') as f:
        f.write('\n' + ' '.join(s) + '\n')
        traceback.print_exc(file=f)

def group_data(data, degree=4, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def cv_loop(X, y, model, N, N_JOBS = 4, seed=25):
    scores = cross_validation.cross_val_score(model, X, y,
            scoring='roc_auc', #score_func = metrics.auc_score,
            pre_dispatch = N_JOBS,
            n_jobs = N_JOBS,
            cv = cross_validation.StratifiedShuffleSplit(y, random_state=seed, n_iter=N))
    return sum(scores) / N
    
def loadData(train='train.csv', test='test.csv', resource=None):

    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    if resource is not None:
        train_data = train_data[train_data.RESOURCE == resource]
        test_data = test_data[test_data.RESOURCE == resource]

    ids = test_data.id
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))

    num_train = np.shape(train_data)[0]
    
    # Transform data
    print "Transforming data..."
    dp = group_data(all_data, degree=2) 
    dt = group_data(all_data, degree=3)
    #dc = group_data(all_data, degree=4)
    #d5 = group_data(all_data, degree=5)

    y = array(train_data.ACTION)
    X = all_data[:num_train]
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]
    #X_4 = dc[:num_train]
    #X_5 = d5[:num_train]

    X_test = all_data[num_train:]
    X_test_2 = dp[num_train:]
    X_test_3 = dt[num_train:]
    #X_test_4 = dc[num_train:]
    #X_test_5 = d5[num_train:]

    X_train_all = np.hstack((X, X_2, X_3)) #, X_4))#, X_5))
    X_test_all = np.hstack((X_test, X_test_2, X_test_3)) #, X_test_4))#, X_test_5))
    num_features = X_train_all.shape[1]
    return X_train_all, y, X_test_all, X_test, num_features, num_train, ids


def greedySelection(Xts=None, y=None, model=None, N=None, seed=None):
    #return [0, 7, 8, 9, 10, 11, 20, 36, 37, 41, 42, 47, 51, 53, 63, 64, 67, 69, 71, 81, 85, 88]

    print "Performing greedy feature selection..."
    score_hist = []
    good_features = set([])
    # Greedy feature selection loop
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        for f in range(len(Xts)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                score = cv_loop(Xt, y, model, N, seed=seed)
                scores.append((score, f))
                #print "Feature: %i Mean AUC: %f" % (f, score)
                print ("%f " % score),
        print
        good_features.add(sorted(scores)[-1][1])
        score_hist.append(sorted(scores)[-1])
        print "Current features: %s" % sorted(list(good_features))
    # Remove last added feature from good_features
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    return good_features
   

def optimizeHyperparameter(model=None, Xts = None, y = None, N = None, features = None, X0 = 2.0):

    Xt = sparse.hstack([Xts[j] for j in features]).tocsr()

    def func(C):
        if C < 0:
            C = -C
        model.C = C
        score = cv_loop(Xt, y, model, N)
        print "C: %f Mean AUC: %f" %(C, score)
        return 1.0 - score

    r, fopt, iter, funcalls, warnflag = \
            fmin(func, X0, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=1, disp=1, retall=0)
    bestC = r[0]
    if bestC < 0: bestC = -bestC
    bestScore = 1.0 - fopt
    return bestC, bestScore

def saveScore(submissionFile, stats, logFile='resources/scores.txt'):
    with open(logFile, 'a') as f:
        stats = ' '.join(str(k)+'='+str(v) for k,v in stats.items())
        f.write('%s %s\n' % (submissionFile, stats))
 
def logistic_regression(seed=25, features=None, data=None, N=20, C=None):
    model = linear_model.LogisticRegression()
    model.predict = lambda M, x: M.predict_proba(x)[:,1]
    
    X_train_all, y, X_test_all, X_test, num_features, num_train, ids = data
    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection 
    Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]

    good_features = features or greedySelection(Xts=Xts, y = y, model = model, N = N, seed = seed)
    print "Selected features %s" % good_features

    score = None
    if C is None:
        C, score = optimizeHyperparameter(model = model, Xts = Xts, y = y, N = N, features = good_features)
    stats = dict(C=C, AUC=score, features=repr(good_features))

    print "Performing One Hot Encoding on entire dataset..."
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = OneHotEncoder(Xt)
    
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]
    print "Training full model..."
    model.fit(X_train, y)
    
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    return ids, preds, "logistic_regression", stats

def create_test_submission(filename, prediction, ids):
    content = ['id,ACTION']
    print 'prediction:', prediction
    for i, p in zip(ids, prediction):
        content.append('%i,%f' % (i,p))
    with open(filename, 'w') as f:
        f.write('\n'.join(content))
    print 'Saved'

def everything(train='data/train.csv', test='data/test.csv', seed = 41, N = 10, data = None):
    ids, preds, algorithmName, stats = logistic_regression(seed=seed, data=data, N=N)
    submissionFile = 'logistic_regression_N=%d_%s_4th.csv' % (N, seed)
    create_test_submission(submissionFile, preds, ids)
    saveScore(submissionFile, stats, logFile='scores_optimized.txt')

def checkSeeds(begin, end, train='data/train.csv', test='data/test.csv', N = 10):
    data = loadData(train = train, test = test)
    #for seed in range(43,55):
    for seed in range(begin,end):
        everything(train = train, test = test, seed = seed, data = data, N = N)

if __name__ == "__main__":
    checkSeeds(56, 65, N = 20)
