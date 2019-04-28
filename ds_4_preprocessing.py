import numpy as np
import pandas as pd

data = pd.read_csv('data/dataset4_v0.csv')
pd.set_option('display.max_columns', 60)
print('Data is loaded')

data = data.drop(('ID'), axis=1)

categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']

print('\nnumerical_columns : ', len(numerical_columns), numerical_columns, '\n')
print(data.describe())
print('\ncategorical_columns : ', len(categorical_columns), categorical_columns, '\n')
print(data[categorical_columns].describe())

print('\nunique values of a categorical feature : \n')
for c in categorical_columns:
    print(len(data[c].unique()), data[c].unique())

print('\nnumber of filled : ', len(data.count(axis=0)), '\n', data.count(axis=0))

data_describe = data.describe(include=[object])
binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print('\nbinary columns : ', len(binary_columns), binary_columns)
print('\nnon-binary columns : ', len(nonbinary_columns), nonbinary_columns, '\n')

for c in binary_columns[0:]:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
print(data[binary_columns].describe())

data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean())/data_numerical.std()
print('\nnormalization\n', data_numerical.describe())

data = pd.concat((data_numerical, data[binary_columns]), axis=1)
data = pd.DataFrame(data, dtype=float)

print('\nCreated a new data #1 with a size of ', data.shape)
data = data.rename(columns={'Churn': 'Target'})
print(data.columns)
print(data.describe())

print('\n', data.shape)
data.to_csv(r'data\dataset4_v1.csv', index=None)
