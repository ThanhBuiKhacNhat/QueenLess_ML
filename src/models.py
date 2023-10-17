from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from .model_distributions import *


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


MODELS = {
    'logistic_regression': {
        'name': 'Logistic Regression',
        'distribution': LOGISTIC_REGRESSION_DIST
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'distribution': KNN_DIST
    },
    'svm': {
        'name': 'Support Vector Machine',
        'distribution': SVM_DIST
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'distribution': DECISION_TREE_DIST
    },
    'random_forest': {
        'name': 'Random Forest',
        'distribution': RANDOM_FOREST_DIST
    },
    'extra_trees': {
        'name': 'Extra Trees',
        'distribution': EXTRA_TREES_DIST
    },
    'xgboost': {
        'name': 'XGBoost',
        'distribution': XGBOOST_DIST
    }
}
