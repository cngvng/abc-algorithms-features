from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from helpers.data_loader import load, get_features_labels
from helpers.plots import roc_curves_plot
from algorithms.abc import ArtificialBeeColony
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import random


def run_compare(path, n_iters=10):
    acc_ABC = list()
    acc_nor = list()
    df = load(path)
    features, labels = get_features_labels(df)
    for i in range(n_iters):
        X_train, X_test, y_train, y_test = get_train_test_data(features, labels, 0.3)
        y_test_binarized = class_binarize(y_test, [1, 2])

        clf = DecisionTreeClassifier()
        clf = train_cart(clf, X_train, y_train)

        # n_classes = y_test_binarized.shape[1]
        y_pred_binarized = class_binarize(test_cart(clf, X_test), [1, 2])
        acc = metrics.accuracy_score(y_test_binarized, y_pred_binarized)
        acc_nor.append(acc)

        modification_rate = 0.3
        cycle = 10
        abc = ArtificialBeeColony(clf, X_train.columns, X_train, X_test, modification_rate)
        accuracy, selected_features, bee, _ = abc.execute(X_train, y_train, X_test, y_test, cycle, acc)
        acc_ABC.append(accuracy)
    
    results = []
    results.append(acc_nor)
    results.append(acc_ABC)

    names = ('Without ABC algorithm', 'With ABC algorithm')
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy')
    plt.savefig('results/compare.png')

# run_compare('/Users/cngvng/Desktop/abc-algorithms-features/data/german.txt')
    

def run_cart(path, test_size):
    df = load(path)
    features, labels = get_features_labels(df)
    X_train, X_test, y_train, y_test = get_train_test_data(features, labels, test_size)
    y_test_binarized = class_binarize(y_test, [1, 2])

    clf = DecisionTreeClassifier()
    clf = train_cart(clf, X_train, y_train)

    n_classes = y_test_binarized.shape[1]
    y_pred_binarized = class_binarize(test_cart(clf, X_test), [1, 2])
    roc_auc_values, fpr, tpr = roc_auc(n_classes, y_test_binarized, y_pred_binarized)
    roc_curves_plot(roc_auc_values, fpr, tpr, n_classes)

def run_cart_abc(path, test_size):
    df = load(path)
    features, labels = get_features_labels(df)
    X_train, X_test, y_train, y_test = get_train_test_data(features, labels, test_size)
    y_test_binarized = class_binarize(y_test, [1, 2])

    clf = DecisionTreeClassifier()
    clf = train_cart(clf, X_train, y_train)

    n_classes = y_test_binarized.shape[1]
    y_pred = test_cart(clf, X_test)
    y_pred_binarized = class_binarize(y_pred, [1, 2])
    score = get_score(clf, X_test, y_test)

    print(reports(y_test, y_pred))

    modification_rate = 0.3
    cycle = 10
    abc = ArtificialBeeColony(clf, X_train.columns, X_train, X_test, modification_rate)
    accuracy, selected_features, bee, _ = abc.execute(X_train, y_train, X_test, y_test, cycle, score)

    print(reports(y_test, bee.get_y_pred()))
    y_pred_binarized = class_binarize(bee.get_y_pred(), [1, 2])
    roc_auc_values, fpr, tpr = roc_auc(n_classes, y_test_binarized, y_pred_binarized)
    roc_curves_plot(roc_auc_values, fpr, tpr, n_classes)

def get_train_test_data(features, classes, test_size, random_state = 0):
    return train_test_split(features, classes, test_size = test_size, stratify=classes, random_state = random.randint(0, 100000))

def train_cart(cart, features, classes):
    return cart.fit(features, classes)

def test_cart(cart, x_test):
    return cart.predict(x_test)

def get_score(clf, x_test, y_true):
    return clf.score(x_test, y_true)

def class_binarize(y, classes):
    return label_binarize(y, classes = classes)

def roc_auc(n_classes, y_test, y_score):
    fpr = dict() # false positive rate
    tpr = dict() # true positive rate
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    return roc_auc, fpr, tpr

def reports(y_true, y_pred):
    return classification_report(y_true, y_pred)