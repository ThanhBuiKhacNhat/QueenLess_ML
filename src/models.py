from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier


def get_model(estimator):
    if estimator == 'logistic_regression':
        return LogisticRegression()

    if estimator == 'knn':
        return KNeighborsClassifier()

    if estimator == 'svm':
        return SVC()

    if estimator == 'decision_tree':
        return DecisionTreeClassifier()

    if estimator == 'random_forest':
        return RandomForestClassifier()

    if estimator == 'extra_trees':
        return ExtraTreesClassifier()

    if estimator == 'xgboost':
        return XGBClassifier()

    raise ValueError('Invalid model name: {}'.format(estimator))


LOGISTIC_REGRESSION_GRID = {
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'max_iter': [500, 1000]
}

KNN_GRID = {
    'n_neighbors': [2 * i + 1 for i in range(10)],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

SVM_GRID = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': ['scale', 'auto']
}

DECISION_TREE_GRID = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

RANDOM_FOREST_GRID = {
    'n_estimators': [10, 50, 100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

EXTRA_TREES_GRID = {
    'n_estimators': [10, 50, 100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

XGBOOST_GRID = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [1, 2, 3, 4, 5],
    'learning_rate': [0.01, 0.1, 1.0],
    'gamma': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0]
}

MODELS = {
    'logistic_regression': {
        'grid': LOGISTIC_REGRESSION_GRID,
        'default': {
            'penalty': 'l2',
            'C': 1.0
        }
    },
    'knn': {
        'grid': KNN_GRID,
        'default': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'p': 2
        }
    },
    'svm': {
        'grid': SVM_GRID,
        'default': {
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale'
        }
    },
    'decision_tree': {
        'grid': DECISION_TREE_GRID,
        'default': {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    },
    'random_forest': {
        'grid': RANDOM_FOREST_GRID,
        'default': {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    },
    'extra_trees': {
        'grid': EXTRA_TREES_GRID,
        'default': {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    },
    'xgboost': {
        'grid': XGBOOST_GRID,
        'default': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'gamma': 0,
            'reg_lambda': 1.0
        }
    }
}
