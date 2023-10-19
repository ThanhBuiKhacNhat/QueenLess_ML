import scipy.stats as st


LOGISTIC_REGRESSION_DIST = {
    'C': st.uniform(loc=0, scale=10),
    'penalty': ['l2'],
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
    'max_iter': [100, 200, 300, 400, 500],
}

KNN_DIST = {
    'n_neighbors': [i for i in range(1, 20, 2)],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40, 50, 60],
    'p': [1, 2]
}

DECISION_TREE_DIST = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

RANDOM_FOREST_DIST = {
    'n_estimators': [50, 100, 200, 300, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

EXTRA_TREES_DIST = {
    'n_estimators': [50, 100, 200, 300, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

XGBOOST_DIST = {
    'n_estimators': [50, 100, 200, 300, 500, 1000],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.3, 1.0],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 4],
}

SVM_DIST = {
    'C': st.expon(scale=1.),
    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    # 'gamma': ['scale', 'auto'] + list(st.expon(scale=.1))
    'gamma': st.expon(scale=.1)
}