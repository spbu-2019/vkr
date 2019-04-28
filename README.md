# Forecasting in the outflow problem

This repository represents the software part of the qualification work Forecasting in the outflow problem. 
The outflow forecasting tool is publicly available due to the increased urgency of the outflow problem and the high popularity of machine learning competitions

## Getting Started

Briefly about how to run the Python file is listed [here](http://www.cs.bu.edu/courses/cs108/guides/runpython.html). It is much easier to create a project in the [IDE](https://www.jetbrains.com/pycharm/) and download files from the [repository](https://github.com/spbu-2019/vkr).

## Built With

* [Python](https://www.python.org/) - Used programming language
* [Scikit-Learn](https://scikit-learn.org/stable/) - Tools for data mining and data analysis
* [PyCharm](https://www.jetbrains.com/pycharm/) - The Python IDE by JetBrains

## Running the tests

* The first step is to run the files related to data preprocessing. The result of their work will be data sets that can be loaded into the tool to solve the outflow problem, and you will see a complete analysis of the available data. Data preprocessing involves specialized work with each data set: for dataset #1 - [ds_1_preprocessing.py](https://github.com/spbu-2019/vkr/blob/master/ds_1_preprocessing.py), for dataset #2 - [ds_2_preprocessing.py](https://github.com/spbu-2019/vkr/blob/master/ds_2_preprocessing.py), for dataset #3 - [ds_3_preprocessing.py](https://github.com/spbu-2019/vkr/blob/master/ds_3_preprocessing.py), for dataset #4 - [ds_4_preprocessing.py](https://github.com/spbu-2019/vkr/blob/master/ds_4_preprocessing.py).

* When you run [universal_search.py](https://github.com/spbu-2019/vkr/blob/master/universal_search.py), you must select one of the six proposed sets (see details in the text of the work), in case of incorrect input of the set marking, the set number 1 is started by default. As a result, you will get estimates of the algorithms and images of ROC-AUC curves. 

* To get the *"Feature Importance"* for all algorithms with bar chart images, run the file [universal_feature_importance.py](https://github.com/spbu-2019/vkr/blob/master/universal_feature_importance.py)



## Examples

The report of each algorithm looks like this:

```
Search Resutls : 
best_estimator :  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=34, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=189, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
best_params :  {'n_estimators': 189, 'min_samples_split': 5, 'max_features': 'auto', 'max_depth': 34}
best score :  0.9183570527832822
AUC : 0.9215594954295006
Accuracy : 0.964
Precision : 0.926829268292683
Recall : 0.7835051546391752
F(1,5) : 0.8226477935054122

Resutls without search for : 
 RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
AUC : 0.9184572393868111
Accuracy : 0.9573333333333334
Precision : 0.922077922077922
Recall : 0.7319587628865979
F(1,5) : 0.7815410668924639 
```
## Authors

* **Pavel Kiva** - *bachelor, Saint-Petersburg State University*

## Acknowledgments

* **Natalia Grafeyeva** - *scientific supervisor, Saint-Petersburg State University*
* See also the references and more details in Graduation Thesis 
