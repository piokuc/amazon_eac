import sys
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations
import numpy as np
import pandas as pd
import sklearn.ensemble #.RandomForestClassifier

SEED = 1234

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    with open(filename, 'w') as f:
        f.write('\n'.join(content))
    print 'Saved'

def cv_loop(X, y, model, N, N_JOBS = None, cv = None):
    if cv is None:
        cv = cross_validation.StratifiedShuffleSplit(y, random_state=SEED, n_iter=N)
    scores = cross_validation.cross_val_score(model, X, y,
            scoring='roc_auc', #score_func = metrics.auc_score,
            pre_dispatch = N_JOBS,
            n_jobs = N_JOBS,
            cv = cv)
    print 'cv_loop => scores =', scores
    return sum(scores) / N
    
def main(train='train.csv', test='test.csv', submit='rf_pred.csv',
        estimators=100):    
    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    
    y_train = array(train_data.ACTION)
    X_train = array(train_data.ix[:,1:-1])
    X_test = array(test_data.ix[:,1:-1]) 

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    model.predict = lambda M, x: M.predict_proba(x)[:,1]
    
    print "Training full model..."
    import sys
    N=10 
    score = cv_loop(X_train, y_train, model, N, N_JOBS = 4)
    print 'AUC =', score
    model.fit(X_train, y_train)
    
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    create_test_submission(submit, preds)
    
if __name__ == "__main__":
    args = dict(train = 'data/train.csv',
             test =     'data/test.csv',
             submit =   'random_forest_pred.csv',
             estimators=len(sys.argv) > 1 and int(sys.argv[1]) or 200,
              )
    main(**args)
