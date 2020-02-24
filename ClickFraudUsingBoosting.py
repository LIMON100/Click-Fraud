import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB , GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import gc



dtypes = {
        'ip'            : 'uint16',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32' 
        }


testing = True
if testing:
    train_path = "G:/Software/Machine learning/1/20. Ensembling/Boosting/train_sample.csv"
    skiprows = None
    nrows = None
    colnames=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
else:
    train_path = "train.csv"
    skiprows = range(1, 144903891)
    nrows = 10000000
    colnames=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']

# read training data
train_sample = pd.read_csv(train_path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=colnames)



dataset = pd.read_csv('G:/Software/Machine learning/1/20. Ensembling/Boosting/train_sample.csv' , dtype = dtypes)
dataset = dataset.drop(['attributed_time'] , axis = 1)


print(dataset.memory_usage())
print('Training dataset uses {0} MB'.format(dataset.memory_usage().sum()/1024**2))



def fraction_unique(x):
    return len(train_sample[x].unique())

number_unique_vals = {x: fraction_unique(x) for x in dataset.columns}
number_unique_vals



plt.figure(figsize=(14, 8))
sns.countplot(x="app", data=dataset)


plt.figure(figsize=(14, 8))
sns.countplot(x="device", data=dataset)



plt.figure(figsize=(14, 8))
sns.countplot(x="channel", data=dataset)


plt.figure(figsize=(14, 8))
sns.countplot(x="os", data=dataset)


100*(train_sample['is_attributed'].astype('object').value_counts()/len(dataset.index))


app_target = dataset.groupby('app').is_attributed.agg(['mean', 'count'])


frequent_apps = dataset.groupby('app').size().reset_index(name = 'count')
frequent_apps = frequent_apps[frequent_apps['count']>frequent_apps['count'].quantile(0.80)]
frequent_apps = frequent_apps.merge(train_sample, on='app', how='inner')


plt.figure(figsize=(10,10))
sns.countplot(y="app", hue="is_attributed", data=frequent_apps)


def timeFeatures(df):
    # Derive new features using the click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    return df


dataset = timeFeatures(train_sample)
dataset.drop(['click_time', 'datetime'], axis=1, inplace=True)




int_vars = ['app', 'device', 'os', 'channel', 'day_of_week','day_of_year', 'month', 'hour']
dataset[int_vars] = dataset[int_vars].astype('uint16')

print(dataset.dtypes)

ip_count = train_sample.groupby('ip').size().reset_index(name='ip_count').astype('int16')
ip_count.head()



def grouped_features(df):
    # ip_count
    ip_count = df.groupby('ip').size().reset_index(name='ip_count').astype('uint16')
    ip_day_hour = df.groupby(['ip', 'day_of_week', 'hour']).size().reset_index(name='ip_day_hour').astype('uint16')
    ip_hour_channel = df[['ip', 'hour', 'channel']].groupby(['ip', 'hour', 'channel']).size().reset_index(name='ip_hour_channel').astype('uint16')
    ip_hour_os = df.groupby(['ip', 'hour', 'os']).channel.count().reset_index(name='ip_hour_os').astype('uint16')
    ip_hour_app = df.groupby(['ip', 'hour', 'app']).channel.count().reset_index(name='ip_hour_app').astype('uint16')
    ip_hour_device = df.groupby(['ip', 'hour', 'device']).channel.count().reset_index(name='ip_hour_device').astype('uint16')
    
    # merge the new aggregated features with the df
    df = pd.merge(df, ip_count, on='ip', how='left')
    del ip_count
    df = pd.merge(df, ip_day_hour, on=['ip', 'day_of_week', 'hour'], how='left')
    del ip_day_hour  
    df = pd.merge(df, ip_hour_channel, on=['ip', 'hour', 'channel'], how='left')
    del ip_hour_channel
    df = pd.merge(df, ip_hour_os, on=['ip', 'hour', 'os'], how='left')
    del ip_hour_os
    df = pd.merge(df, ip_hour_app, on=['ip', 'hour', 'app'], how='left')
    del ip_hour_app
    df = pd.merge(df, ip_hour_device, on=['ip', 'hour', 'device'], how='left')
    del ip_hour_device
    
    return df



dataset = grouped_features(dataset)
print('Training dataset uses {0} MB'.format(dataset.memory_usage().sum()/1024**2))

gc.collect()



x = dataset.drop('is_attributed', axis=1)
y = dataset[['is_attributed']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=90)


adt = DecisionTreeClassifier(max_depth = 5)
adt1 = AdaBoostClassifier(base_estimator = adt , n_estimators = 100 , learning_rate = 0.1 , algorithm = "SAMME")


adt1.fit(x_train , y_train)
prediction = adt1.predict_proba(x_test)

a = metrics.roc_auc_score(y_test, prediction[:,1])



param_grids = {'base_estimator__max_depth': [2 , 3 , 5],
               'n_estimators': [100 , 200 , 400 , 500],
               'learning_rate': [0.6 , 1.0 , 1.5 , 2.1]
               }


dt = DecisionTreeClassifier()
abc = AdaBoostClassifier(base_estimator = dt , algorithm = "SAMME")

folds = 3
grd_search = GridSearchCV(abc , cv = folds , param_grid = param_grids , scoring = 'roc_auc' , return_train_score = True , verbose = 1)

grd_search.fit(x_train , y_train)
cv_results = pd.DataFrame(grd_search.cv_results_)
cv_results


plt.figure(figsize=(16,6))
for n, depth in enumerate(param_grids['base_estimator__max_depth']):
    

    # subplot 1/n
    plt.subplot(1,3, n+1)
    depth_df = cv_results[cv_results['param_base_estimator__max_depth']==depth]

    plt.plot(depth_df["param_n_estimators"], depth_df["mean_test_score"])
    plt.plot(depth_df["param_n_estimators"], depth_df["mean_train_score"])
    plt.xlabel('n_estimators')
    plt.ylabel('AUC')
    plt.title("max_depth={0}".format(depth))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
    
    


model = XGBClassifier()
model.fit(x_train , y_train)

prediction1 = model.predict_proba(x_test)
print(prediction1[:10])



roc = metrics.roc_auc_score(y_test , prediction1[: , 1])
print("AUC: %.2f%%" % (roc*100.0))



folds = 3

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


xgb_model = XGBClassifier(max_depth=2, n_estimators=200)
model_cv = GridSearchCV(estimator = xgb_model, param_grid = param_grid, scoring= 'roc_auc', cv = folds, verbose = 1 , return_train_score=True)      




folds = 3

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


xgb_model = XGBClassifier(max_depth=2, n_estimators=200)
model_cv = GridSearchCV(estimator = xgb_model,  param_grid = param_grid, scoring= 'roc_auc', cv = folds, verbose = 1,return_train_score=True)     

model_cv.fit(x_train, y_train)  

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_max_depth'] = cv_results['param_max_depth'].astype('float')
cv_results.head()

plt.figure(figsize=(16,6))


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')




params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.6,
         'objective':'binary:logistic'}

# fit model on training data
model = XGBClassifier(params = params)
model.fit(x_train, y_train)



y_pred = model.predict_proba(x_test)
auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
print(auc)

plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()


print(confusion_matrix(y_test , model.predict(x_test)))


from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_lr = lr.predict_proba(x_test)

auc_lr = metrics.roc_auc_score(y_test, y_lr[:, 1])
print(auc_lr)


lr = SGDClassifier(class_weight='balanced', penalty='l2', loss = 'log' ,  random_state=42)
lr.fit(x_train, y_train)

ccv_lr = CalibratedClassifierCV(lr , method = "sigmoid")
ccv_lr.fit(x_train , y_train)

y_lr = ccv_lr.predict_proba(x_test)


auc_lr = metrics.roc_auc_score(y_test, y_lr[:, 1])
print(auc_lr)


svc = SVC()
svc.fit(x_train , y_train)
y_svc =  svc.predict_proba(x_test)



mnb = MultinomialNB()
mnb.fit(x_train , y_train)
y_mnb = mnb.predict_proba(x_test)


auc_mnb = metrics.roc_auc_score(y_test, y_mnb[:, 1])
print(auc_mnb)


rf = RandomForestClassifier(n_estimators = 600 , max_depth = 5)
rf.fit(x_train , y_train)
y_rf = rf.predict_proba(x_test)

auc_mnb = metrics.roc_auc_score(y_test , y_rf[:, 1])
print(auc_mnb)