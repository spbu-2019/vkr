import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/dataset1_v0.csv')
pd.set_option('display.max_columns', 60)
print('Data is loaded')

categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']

print('\nnumerical_columns : ', len(numerical_columns), numerical_columns, '\n')
print(data.describe())
print('\ncategorical_columns : ', len(categorical_columns), categorical_columns, '\n')
print(data[categorical_columns].describe())

print('\nunique values of a categorical feature : \n')
for c in categorical_columns:
    print(c, len(data[c].unique()), data[c].unique())

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

data_encoder = data[nonbinary_columns]
data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print('\nconverting non-binary to a set of binary ', len(data_nonbinary.columns), '\n', data_nonbinary.columns)

encoder = LabelEncoder()
for c in nonbinary_columns[0:]:
    encoder.fit(data_encoder[c])
    data_encoder[c] = encoder.transform(data_encoder[c])
print('\nconverting non-binary with encoder\n', data_encoder.describe())

data_encoder = (data_encoder - data_encoder.mean())/data_encoder.std()

data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean())/data_numerical.std()
print('\nnormalization\n', data_numerical.describe())

data1 = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data1 = pd.DataFrame(data1, dtype=float)
data1 = data1.drop(('StandardHours'), axis=1)
data1 = data1.drop(('EmployeeCount'), axis=1)

data2 = pd.concat((data_numerical, data[binary_columns], data_encoder), axis=1)
data2 = pd.DataFrame(data2, dtype=float)
data2 = data2.drop(('StandardHours'), axis=1)
data2 = data2.drop(('EmployeeCount'), axis=1)

print('\nCreated a new data #1 with a size of ', data1.shape)
data1 = data1.rename(columns={'Attrition': 'Target'})
print(data1.columns)
print(data1.describe())

print('\nCreated a new data #2 with a size of ', data2.shape)
data2 = data2.rename(columns={'Attrition': 'Target'})
print(data2.columns)
print(data2.describe())

print('\n', data1.shape, data2.shape)
data1.to_csv(r'data\dataset1_v1.csv', index=None)
data2.to_csv(r'data\dataset1_v2.csv', index=None)

