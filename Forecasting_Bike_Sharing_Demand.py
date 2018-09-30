# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('C:/Users/yimen/OneDrive/Documents/Python Scripts/input/train.csv')
test = pd.read_csv('C:/Users/yimen/OneDrive/Documents/Python Scripts/input/test.csv')

from scipy.stats import skew
from scipy.stats import norm
from scipy.special import boxcox1p, inv_boxcox

from datetime import date, datetime
import time

### No missing values
### Feature selection
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
print(train.isna().sum().sum())
print(test.isna().sum().sum())

train.head()

train.info()

train.shape

train.describe()

def transform_time(d):
    d['time'] = d['datetime'].apply(lambda x: x[11:13])
    d['date'] = d['datetime'].apply(lambda x: x[:10])
    d['weekday'] = d['date'].apply(lambda s: date(*(int(i) for i in s.split('-'))).weekday() + 1)
    d['month'] = d['date'].apply(lambda s: s[5:7])
    d['day'] = d['date'].apply(lambda s: s[8:10])
    d['year'] = d['date'].apply(lambda s: s[:4])
    # Monday -> 0, Sunday -> 6, so weekday() +1

transform_time(train)
transform_time(test)
data = pd.concat([train[test.columns], test]).reset_index(drop=True)

# help(date(*[int(i) for i in data.loc[1, 'date'].split('-')]).weekday)
# date(*[int(i) for i in data.loc[1, 'date'].split('-')]).weekday()

# Non-linear data correlation
plt.subplots(figsize=(12, 9))
sns.heatmap(train.corr(), square=True, annot=True, fmt='.2f')

from sklearn.feature_selection import mutual_info_regression
temp_train = train.drop(['datetime', 'date'], axis=1)
mutual_res = mutual_info_regression(temp_train, train['count'])

pd.Series(mutual_res, index=temp_train.columns + "~count").sort_values(ascending=False)

sns.distplot(train['count'])

# Box-Cox Transformation can make it 0.37, which minimize the skew, 
# but I still use the log transformation, since it's more reasonable in loss function
sns.distplot(boxcox1p(train.loc[:, 'count'], 0.37), fit=norm)

print(skew(train['count']), skew(np.log2(train['count'] + 1)),
      skew(boxcox1p(train.loc[:, 'count'], 0.37)), sep='\t')

train['count'] = np.log2(train['count'] + 1)
# train['count'] = boxcox1p(train['count'], 0.37)

sns.distplot(train['count'], fit=norm)

d = train['windspeed']
(((d - d.mean()) / d.std()) ** 3).mean()

# Continuous feature
sns.heatmap(train[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr(), fmt='.4f', annot=True, square=True)

value_feature = ['temp', 'atemp', 'humidity', 'windspeed']
skew(train[value_feature])

sns.distplot(train['windspeed'])

plt.scatter(data['atemp'], data['temp'])
plt.xlabel('atemp')
plt.ylabel('temp')

(data['temp'] - data['atemp']).sort_values(ascending=False)[:30]

plt.subplot(121)
data.loc[10420:10460, 'atemp'].plot()
plt.title('atemp')
plt.subplot(122)
data.loc[10420:10460, 'temp'].plot()
plt.title('temp')

# Delete some abnormal data
sel = (data['temp'] - data['atemp']).sort_values(ascending=False)[:24].index
data.loc[sel, 'atemp'] = data.loc[sel, 'temp']
plt.scatter(data['atemp'], data['temp'])


# Discrete featureS -- Box diagram

# plt.scatter('season', 'count', data=train)
sns.boxplot('season', 'count', data=train)

sns.boxplot(train['time'], train['count'])

sns.boxplot(train['weekday'], train['count'])

sns.boxplot(train['holiday'], train['count'])

pd.crosstab(data.loc[data['workingday'] == 0, 'holiday'], data.loc[data['workingday'] == 0, 'weekday'])

pd.crosstab(data.loc[data['workingday'] == 1, 'holiday'], data.loc[data['workingday'] == 1, 'weekday'])
#  When workingday = 0，holiday = 1; vice versa  

plt.subplots(figsize=(15,3))
plt.subplot(131)
sns.boxplot(train['season'], np.log(train['count']))
plt.subplot(132)
sns.boxplot(train['month'], train['count'])
plt.subplot(133)
sns.boxplot(train['day'], train['count'])

sns.boxplot(train['year'], train['count'])

sns.distplot(data['windspeed'])

data = data.drop(['day', 'date', 'datetime', 'atemp'], axis=1)

data.info()

class_feature = ['weather', 'time', 'weekday', 'month', 'year', 'season']
data = pd.get_dummies(data, columns=class_feature)

data.info()

data.head()

### Begin to train

train_X = data[:train.shape[0]]
test_X = data[train.shape[0]:]
train_y = train['count']

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler # Normalization

def rmse_cv(model):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv = kf))  # 默认的cv没有shuffle
    return(rmse.mean())
    
c = train_X.columns

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

pd.DataFrame(train_X, columns=c).head()

### Fitting models
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

rmse_cv(KNeighborsRegressor())

dt = DecisionTreeRegressor()
rmse_cv(dt)

### Variance reduction
# Hyperparameters tuning through Grid Search
gs = GridSearchCV(DecisionTreeRegressor(), scoring="neg_mean_squared_error", cv=3, verbose=1,
                  param_grid={"max_depth": [2, 5, 10, 20, 50, 100, None], 
                              "min_samples_split":[2, 5, 10, 20, 50, 100]}, )
gs.fit(train_X, train_y)
gs.best_params_

gs = GridSearchCV(DecisionTreeRegressor(), scoring="neg_mean_squared_error", cv=3, verbose=1,
                  param_grid={"max_depth": [20, 50, 100, 150, 200, None], 
                              "min_samples_split":[i for i in range(15, 35)]}, )
gs.fit(train_X, train_y)
gs.best_params_

rmse_cv(gs.best_estimator_)

gs.best_estimator_.fit(train_X, train_y)
predict_y = gs.best_estimator_.predict(test_X)

res = pd.DataFrame([test['datetime'].values, 2 ** predict_y - 1], index=['datetime', 'count']).T
res.to_csv('res2.csv', index=False, header=True)

### Some other models
# Polynomial kernel regression & L2 regression
from sklearn.kernel_ridge import KernelRidge
krr = KernelRidge(alpha=1, kernel='polynomial', degree=3)
rmse_cv(krr)

# Ensemble methods
from sklearn.ensemble import GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, loss='huber')
rmse_cv(GBoost)

import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.903, gamma=0.048, learning_rate=0.05, max_depth=5, 
                             min_child_weight=0.7817, n_estimators=3000, reg_alpha=0.7640, reg_lambda=0.8571,
                             subsample=0.8213, silent=1, random_state =7, n_jobs = -1)

xgb.XGBRegressor
xgb.XGBClassifier
rmse_cv(model_xgb)

import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.903, gamma=0.048, learning_rate=0.05, max_depth=5, 
                             min_child_weight=0.7817, n_estimators=3000, reg_alpha=0.7640, reg_lambda=0.8571,
                             subsample=0.8213, silent=1, random_state =7, n_jobs = -1)

xgb.XGBRegressor
xgb.XGBClassifier
rmse_cv(model_xgb)
