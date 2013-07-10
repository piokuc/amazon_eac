from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
import sklearn.ensemble #.RandomForestClassifier
from scipy import sparse
from itertools import combinations
import scipy.optimize

import numpy as np
import pandas as pd

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

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

def cv_loop(X, y, model, N, N_JOBS = 4, SEED=25):
    scores = cross_validation.cross_val_score(model, X, y,
            scoring='roc_auc', #score_func = metrics.auc_score,
            pre_dispatch = N_JOBS,
            n_jobs = N_JOBS,
            cv = cross_validation.StratifiedShuffleSplit(y, random_state=SEED, n_iter=N))
    return sum(scores) / N
    
def loadData(train='train.csv', test='test.csv'):

    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
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
    return X_train_all, y, X_test_all, X_test, num_features, num_train


def greedySelection(Xts=None, y=None, model=None, N=None, SEED=None):
    return [0, 7, 8, 9, 36, 41, 42, 47, 52, 53, 61, 63, 64, 67, 69, 85]
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
                score = cv_loop(Xt, y, model, N, SEED=SEED)
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
   

def optimizeHyperparameter(model=None, Xts = None, y = None, N = None, features = None):
    print "Performing hyperparameter selection..."
    # Hyperparameter selection loop
    score_hist = []
    Xt = sparse.hstack([Xts[j] for j in features]).tocsr()

    for C in np.logspace(1, 4, 20, base=2):
        model.C = C
        score = cv_loop(Xt, y, model, N)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestScore, bestC = sorted(score_hist)[-1]
    print "Best C value: %f" % (bestC)
    return bestC, bestScore


def optimizeHyperparameter(model=None, Xts = None, y = None, N = None, features = None, X0 = 2.0):

    Xt = sparse.hstack([Xts[j] for j in features]).tocsr()

    def func(C):
        if C < 0:
            C = -C
        model.C = C
        score = cv_loop(Xt, y, model, N)
        print "C: %f Mean AUC: %f" %(C, score)
        return 1.0 - score

    #xopt, fopt, iter, funcalls, warnflag, allvecs = \
    r, fopt, iter, funcalls, warnflag = scipy.optimize.fmin(func, X0, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=1, disp=1, retall=0, callback=None)
    bestC = r[0]
    if bestC < 0: bestC = -bestC
    bestScore = 1.0 - fopt
    return bestC, bestScore
     
 
def main(train='train.csv', test='test.csv', submit='logistic_pred.csv', SEED=25, features=None, data=None):
    X_train_all, y, X_test_all, X_test, num_features, num_train = data#loadData(train=train, test=test)
    
    model = linear_model.LogisticRegression()
    model.predict = lambda M, x: M.predict_proba(x)[:,1]
    
    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection 
    Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]

    N = 15
    good_features = features or greedySelection(Xts=Xts, y = y, model = model, N = N, SEED = SEED)
    print "Selected features %s" % good_features

    bestC, bestScore = optimizeHyperparameter(model=model, Xts = Xts, y = y, N = N, features = good_features)

    with open('scores_optimized.txt', 'a') as f:
        f.write('%s: C=%f AUC=%f %s\n' % (submit, bestC, bestScore, repr(good_features)))

    print "Performing One Hot Encoding on entire dataset..."
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]
    
    print "Training full model..."
    model.fit(X_train, y)
    
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    create_test_submission(submit, preds)


good_features = [ #(0, [0, 7, 8, 9, 10, 11, 20, 36, 37, 41, 42, 47, 51, 53, 63, 64, 67, 69, 71, 81, 85, 88]),
#(1, [0, 7, 8, 9, 24, 36, 37, 41, 42, 47, 52, 53, 60, 61, 63, 64, 67, 69, 82, 85]),
#(2, [0, 7, 8, 9, 10, 11, 24, 33, 36, 37, 38, 41, 42, 43, 47, 51, 53, 60, 61, 63, 64, 67, 69, 71, 82, 85]),
#(3, [0, 8, 9, 34, 36, 37, 38, 41, 42, 43, 47, 51, 53, 54, 63, 64, 65, 66, 67, 69, 71, 72, 81, 85]),
#(4, [0, 7, 8, 9, 36, 41, 42, 47, 52, 53, 61, 63, 64, 67, 69, 85]),
#(5, [0, 5, 8, 9, 13, 36, 37, 38, 41, 42, 47, 51, 53, 55, 60, 61, 63, 64, 66, 67, 69, 71, 85]),
#(6, [0, 8, 10, 12, 13, 14, 24, 36, 37, 38, 41, 42, 47, 53, 60, 63, 64, 65, 67, 69, 82, 85]),
#(7, [0, 2, 9, 35, 36, 38, 42, 45, 47, 53, 57, 59, 63, 64, 67, 69, 75, 85]),
#(8, [0, 7, 8, 9, 10, 24, 27, 36, 37, 38, 41, 42, 47, 53, 54, 61, 63, 64, 67, 69, 71, 83, 85, 91]),
#(9, [0, 2, 7, 8, 10, 36, 37, 42, 43, 47, 53, 64, 65, 66, 67, 69, 71, 82, 85]),
#(10, [0, 3, 8, 9, 20, 36, 38, 42, 47, 53, 61, 64, 67, 69, 81, 85, 88]),
#(11, [0, 7, 10, 27, 36, 37, 42, 43, 47, 53, 57, 63, 64, 66, 67, 69, 81, 85]),
#(12, [0, 8, 10, 36, 37, 38, 41, 42, 43, 47, 51, 53, 61, 63, 64, 67, 69, 85]),
#(13, [0, 7, 8, 9, 41, 42, 47, 53, 63, 64, 67, 69, 85]),
#(14, [0, 7, 8, 9, 20, 27, 36, 37, 41, 42, 47, 49, 51, 53, 61, 63, 64, 66, 67, 69, 85, 88]),
#(15, [0, 36, 37, 41, 42, 43, 47, 53, 54, 61, 64, 67, 69, 85]),
#(16, [0, 8, 9, 36, 37, 41, 42, 43, 47, 53, 57, 64, 66, 67, 69, 85]),
#(17, [0, 7, 8, 9, 13, 20, 27, 36, 37, 42, 47, 53, 63, 64, 67, 69, 71, 85, 88]),
#(18, [0, 1, 8, 9, 10, 14, 19, 20, 36, 37, 38, 41, 42, 43, 47, 52, 53, 63, 64, 66, 67, 69, 70, 71, 75, 85, 91]),
#(19, [0, 7, 8, 9, 10, 36, 37, 38, 39, 41, 42, 47, 49, 51, 53, 60, 61, 63, 64, 67, 69, 71, 72, 79, 85, 88]),
(20, [0, 8, 9, 10, 36, 37, 38, 40, 41, 42, 47, 49, 53, 64, 65, 66, 67, 69, 77, 85, 86]),
]
    
if __name__ == "__main__":
    train='data/train.csv'
    test='data/test.csv'
    data = loadData(train=train, test=test)
    for seed, features in good_features:
        main(train=train, test=test, submit='logistic_regression_pred_opt_%d.csv' % seed, SEED=seed, features = features, data=data)
