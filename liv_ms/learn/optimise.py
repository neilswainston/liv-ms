'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
import sys

from rdkit.Chem.rdMolDescriptors import \
    GetHashedTopologicalTorsionFingerprintAsBitVect
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing.data import StandardScaler

from liv_ms.chem import encode_fngrprnt
from liv_ms.learn import k_fold, rt
from liv_ms.utils import to_str
import numpy as np


def optimise(X, y, estimator, param_distributions,
             n_iter=16, cv=16, verbose=2, n_jobs=-1):
    '''Hyperparameter optimisation.'''
    rand_search = RandomizedSearchCV(estimator=estimator,
                                     param_distributions=param_distributions,
                                     n_iter=n_iter, cv=cv, random_state=42,
                                     verbose=verbose,
                                     n_jobs=n_jobs,
                                     error_score='raise')

    rand_search.fit(X, y)

    return rand_search


def optimise_rf(X, y, n_iter=16, cv=16, verbose=2, n_jobs=-1):
    '''Random Forest hyperparameter optimisation.'''

    # Create the random grid
    param_distributions = {
        'n_estimators': [int(x) for x in np.linspace(100, 1000, 10)],
        'max_features': ['auto', 'log2', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 100, 10)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'bootstrap': [True, False]
    }

    estimator = RandomForestRegressor()

    return optimise(X, y, estimator, param_distributions,
                    n_iter=n_iter, cv=cv, verbose=verbose, n_jobs=n_jobs)


def _report(cv_results, n_top=3):
    '''Report.'''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(cv_results['rank_test_score'] == i)

        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'
                  .format(cv_results['mean_test_score'][candidate],
                          cv_results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(cv_results['params'][candidate]))
            print('')


def main(args):
    '''main method.'''
    filename = args[0]
    regenerate_stats = bool(int(args[1]))
    verbose = int(args[2])
    n_iter = 12
    cv = 8
    n_jobs = 4
    scaler_func = StandardScaler
    max_rt = 30.0
    stats_df, X, y, y_scaler = rt.get_data(filename, regenerate_stats,
                                           scaler_func=scaler_func,
                                           max_rt=max_rt)

    fngrprnt_func = GetHashedTopologicalTorsionFingerprintAsBitVect
    fngrprnt_enc = np.array([encode_fngrprnt(s, fngrprnt_func)
                             for s in stats_df['smiles']])
    X = np.concatenate([X, fngrprnt_enc], axis=1)
    X = scaler_func().fit_transform(X)

    rand_search = optimise_rf(
        X, y, n_iter=n_iter, cv=cv, verbose=verbose, n_jobs=n_jobs)

    _report(rand_search.cv_results_, n_top=n_iter)

    title = '%s_%s' % (rand_search._estimator_type, to_str(fngrprnt_func))

    k_fold(X, y, rand_search.best_estimator_, title, y_scaler, k=cv,
           do_fit=True)


if __name__ == '__main__':
    main(sys.argv[1:])
