import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import warnings

n_estimators = [int(x) for x in np.linspace(start=1, stop=200, num=100)]
warnings.filterwarnings("ignore")


def search_feature_importance(data, number):
    y = data['Target']
    X = data.drop(('Target'), axis=1)
    print('\n#', number, ' Dataset size : \n', X.shape, y.shape)
    pd.set_option('display.max_columns', 60)
    print('\nData is loaded\n\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    rf = RandomForestClassifier()
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 45, num=45)]
    min_samples_split = [5, 10]
    param = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split}
    rand = RandomizedSearchCV(rf, param_distributions=param, n_iter=10, cv=5,
                              n_jobs=-1, scoring='roc_auc')

    rand.fit(X_train, y_train.values.ravel())
    importances = rand.best_estimator_.feature_importances_
    feature_list = list(X.columns)
    feature_importance = sorted(zip(importances, feature_list), reverse=True)
    df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
    importance = list(df['importance'])
    feature = list(df['feature'])
    print(df)

    plt.style.use('bmh')
    x_values = list(range(len(feature_importance)))
    plt.figure(figsize=(15, 10))
    plt.bar(x_values, importance, orientation='vertical')
    plt.xticks(x_values, feature, rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()


data_1a = pd.read_csv('data/dataset1_v1.csv')
search_feature_importance(data_1a, '1a')

data_1b = pd.read_csv('data/dataset1_v2.csv')
search_feature_importance(data_1b, '1b')

data_2a = pd.read_csv('data/dataset2_v1.csv')
search_feature_importance(data_2a, '2a')

data_2b = pd.read_csv('data/dataset2_v2.csv')
search_feature_importance(data_2b, '2b')

data_3 = pd.read_csv('data/dataset3_v1.csv')
search_feature_importance(data_3, '3')

data_4 = pd.read_csv('data/dataset4_v1.csv')
search_feature_importance(data_4, '4')

