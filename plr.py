from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model, svm
import sklearn.ensemble #.RandomForestClassifier
from scipy import sparse
from itertools import combinations
from scipy.optimize import fmin

import numpy as np
import pandas as pd

from collections import Counter
import sys, traceback

def log_exception(*s):
    with open('exceptions.txt','a') as f:
        f.write('\n' + ' '.join(s) + '\n')
        traceback.print_exc(file=f)

def group_data(data, degree=4, hash=hash, threshold = 3):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    count = Counter()
    for indicies in combinations(range(n), degree):
        for v in data[:,indicies]:
            h = hash(tuple(v))
            count[h] += 1
    for indicies in combinations(range(n), degree):
        row = []
        for v in data[:,indicies]:
            h = hash(tuple(v))
            cnt = count[h]
            if cnt < threshold:
                row.append('rare')
            else:
                row.append(h)
        new_data.append(row)
        #new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    r = array(new_data).T
    print 'group_data: =>', r.shape
    return r

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
    
def loadData(train='train.csv', test='test.csv', degree = 4, threshold = 3):

    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    ids = test_data.id
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
    num_train = np.shape(train_data)[0]
    
    print "Transforming data..."

    y = array(train_data.ACTION)

    X = all_data[:num_train]
    X_test = all_data[num_train:]

    X_train_all = [ X ]
    X_test_all = [ X_test ]
    for i in range(2, degree + 1):
        d = group_data(all_data, degree = i, threshold = threshold)
        X_train_all.append(d[:num_train])
        X_test_all.append(d[num_train:])

    X_train_all = np.hstack(tuple(X_train_all))
    X_test_all = np.hstack(tuple(X_test_all))

    num_features = X_train_all.shape[1]
    return X_train_all, y, X_test_all, X_test, num_features, num_train, ids


def greedySelection(Xts=None, y=None, model=None, N=None, seed=None):

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

def saveScore(submissionFile, stats, logFile='scores.txt'):
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

def everything(train='data/train.csv', test='data/test.csv', seed = 41, N = 10, data = None, degree=4, threshold=3):
    ids, preds, algorithmName, stats = logistic_regression(seed=seed, data=data, N=N)
    submissionFile = 'lr_rare_events_degree=%d_N=%d_seed=%d_threshold=%d.csv' % (degree, N, seed, threshold)
    create_test_submission(submissionFile, preds, ids)
    saveScore(submissionFile, stats, logFile='scores_optimized.txt')

def checkSeeds(begin, end, train='data/train.csv', test='data/test.csv', threshold=3, degree=3, N = 10):
    data = loadData(train = train, test = test, threshold = threshold)
    for seed in range(begin,end):
        everything(train = train, test = test, seed = seed, data = data, degree = degree, threshold = threshold, N = N)

if __name__ == "__main__":
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    degree = 4
    if len(sys.argv) == 4: degree = int(sys.argv[3])
    threshold=3
    if len(sys.argv) == 5: threshold = int(sys.argv[4])

    print "check seeds from %(begin) to %(end), degree=%(degree),threshold=%(threshold)" % locals()

    checkSeeds(begin, end, degree=degree, threshold=threshold, N = 10)
