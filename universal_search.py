import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, accuracy_score, recall_score, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import uniform
import warnings

number = input("enter the data number (options: 1a, 1b, 2a, 2b, 3, 4) ")
path_to_data = {
    '1a': 'data/dataset1_v1.csv',
    '1b': 'data/dataset1_v2.csv',
    '2a': 'data/dataset2_v1.csv',
    '2b': 'data/dataset2_v2.csv',
    '3': 'data/dataset3_v1.csv',
    '4': 'data/dataset4_v1.csv'
}

try:
    path = path_to_data[number]
except KeyError as e:
    path = 'data/dataset1_v1.csv'
    print('\nWARNING : Incorrect input, work with data #1\n\n')

warnings.filterwarnings("ignore")

data = pd.read_csv(path)
y = data['Target']
X = data.drop(('Target'), axis=1)
print('\n#', number, ' Dataset size : \n', X.shape, y.shape)
pd.set_option('display.max_columns', 60)
print('\nData is loaded\n\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
n_estimators = [int(x) for x in np.linspace(start=1, stop=200, num=100)]


def search(estimator, param):
    rand = RandomizedSearchCV(estimator, param, cv=5, n_iter=10, n_jobs=-1, scoring='roc_auc', verbose=1)
    rand.fit(X_train, y_train.values.ravel())
    print('\nSearch Resutls : \nbest_estimator : ', rand.best_estimator_)
    print('best_params : ', rand.best_params_)
    print('best score : ', rand.best_score_)

    y_auc = rand.predict_proba(X_test)[:, 1]
    y_pred = rand.predict(X_test)
    y_pred_model = rand.predict_proba(X_test)[:, 1]
    fpr_model, tpr_model, _ = roc_curve(y_test, y_pred_model)
    roc_auc_model = roc_auc_score(y_test, y_pred_model)

    s = estimator.__class__.__name__ + ' (AUC = ' + '{:.3f}'.format(roc_auc_model) + ')'
    plt.plot(fpr_model, tpr_model, label=s)

    print('AUC : ' + str(roc_auc_score(y_test, y_auc)))
    print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
    print('Precision : ' + str(precision_score(y_test, y_pred)))
    print('Recall : ' + str(recall_score(y_test, y_pred)))
    print('F(1,5) : ' + str(fbeta_score(y_test, y_pred, beta=1.5)))


def do_simple(estimator):
    estimator.fit(X_train, y_train)
    y_pred_auc = estimator.predict_proba(X_test)[:, 1]
    print('\nResutls without search for : \n', estimator)
    y_pred = estimator.predict(X_test)
    print('AUC : ' + str(roc_auc_score(y_test, y_pred_auc)))
    print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
    print('Precision : ' + str(precision_score(y_test, y_pred)))
    print('Recall : ' + str(recall_score(y_test, y_pred)))
    print('F(1,5) : ' + str(fbeta_score(y_test, y_pred, beta=1.5)), '\n\n')


def do_knn():
    knn = KNeighborsClassifier()
    k_range = list(range(1, 25))
    p_range = list(range(1, 8))
    weight_options = ['uniform', 'distance']
    param = dict(n_neighbors=k_range, weights=weight_options, p=p_range)
    search(knn, param)
    do_simple(knn)


def do_rf():
    rf = RandomForestClassifier()
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 45, num=45)]
    min_samples_split = [5, 10]
    param = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split}
    search(rf, param)
    do_simple(rf)


def do_grd():
    grd = GradientBoostingClassifier()
    loss = ['deviance', 'exponential']
    max_features = ['auto', 'sqrt']
    learning_rate = [0.1, 0.3, 0.5, 0.8, 1.0]
    max_depth = [int(x) for x in np.linspace(1, 30, num=20)]
    min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
    min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
    param = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'loss': loss,
             'learning_rate': learning_rate,
             'min_samples_leaf': min_samples_leaf}
    search(grd, param)
    do_simple(grd)


def do_gnb():
    dtc = GaussianNB()
    dtc.fit(X_train, y_train)
    y_pred_dtc = dtc.predict_proba(X_test)[:, 1]
    fpr_dtc, tpr_dtc, _ = roc_curve(y_test, y_pred_dtc)
    roc_auc_dtc = roc_auc_score(y_test, y_pred_dtc)
    s = dtc.__class__.__name__ + ' (AUC = ' + '{:.3f}'.format(roc_auc_dtc) + ')'
    plt.plot(fpr_dtc, tpr_dtc, label=s)
    print(dtc)
    y_dtc = dtc.predict(X_test)
    print('AUC : ' + str(roc_auc_score(y_test, y_pred_dtc)))
    print('Accuracy : ' + str(accuracy_score(y_test, y_dtc)))
    print('Precision : ' + str(precision_score(y_test, y_dtc)))
    print('Recall : ' + str(recall_score(y_test, y_dtc)))
    print('F(1,5) : ' + str(fbeta_score(y_test, y_dtc, beta=1.5)), '\n\n')


def do_abc():
    abc = AdaBoostClassifier()
    learning_rate = [0.1, 0.5, 1.0]
    param = {'n_estimators': n_estimators,
             'learning_rate': learning_rate}
    search(abc, param)
    do_simple(abc)


def do_LG():
    LG = LogisticRegression(solver='liblinear')
    penalty = ['l1', 'l2']
    C = uniform(loc=0, scale=4)
    param = {'penalty': penalty,
             'C': C}
    search(LG, param)
    do_simple(LG)


def do_rez():
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    do_gnb()
    do_LG()
    do_knn()
    do_rf()
    do_grd()
    do_abc()

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


do_rez()

