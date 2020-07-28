import os
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import sklearn
import xgboost as xgb
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import Callback

Working_path = r'C:\Users\97254\Desktop\Kaggle\Fraud'

# ------------ load & prepar db --------------- #
train_transaction = pd.read_csv(Working_path+'\\train_transaction.csv')
train_identity = pd.read_csv(Working_path+'\\train_identity.csv')
test_transaction = pd.read_csv(Working_path+'\\test_transaction.csv')
test_identity = pd.read_csv(Working_path+'\\test_identity.csv')
sub = pd.read_csv(Working_path+'\\sample_submission.csv')

# --- merge two sources - transaction and identity --- #
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

del train_transaction
del train_identity
del test_transaction
del test_identity


# --- reduce memory usage --- #
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100*mem_usg/start_mem_usg,"% of the initial size")
    return props


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# ------------ clean db --------------- #
# --- remove columns with too many missing values --- #
threshold_to_delete = 25
null_percent = train.isnull().sum()/train.shape[0]*100
cols_to_drop = np.array(null_percent[null_percent > threshold_to_delete].index)

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

# --- complect other missing values --- #
null_percent = test.isnull().sum()/train.shape[0]*100
coll_to_complect = np.array(null_percent[null_percent > 0].index)

# --- since categorical values is missing we will complete with the mode --- #
for i in coll_to_complect:
    print('data type of {} is {}'.format(i, str(train[i].dtype)))
    train[i] = train[i].replace(np.nan, train[i].mode()[0])
    test[i] = test[i].replace(np.nan, train[i].mode()[0])

# --- encoding categorical features --- #
X = train.drop('isFraud', axis=1)
y = train['isFraud']

cat_data = X.select_dtypes(include='object')
num_data = X.select_dtypes(exclude='object')

cat_cols = cat_data.columns.values
num_cols = num_data.columns.values

for i in tqdm(cat_cols): 
    label = LabelEncoder()
    label.fit(list(X[i].values)+list(test[i].values))
    X[i] = label.transform(list(X[i].values))
    test[i] = label.transform(list(test[i].values))

X = X.drop('TransactionID', axis=1)
test = test.drop('TransactionID', axis=1)

# --- delete numeric features that too correlated with each other ---#
Min_corr_to_delete = 0.9
c = X.corr()
col_corr = set()
for i in range(len(c.columns)):
    for j in range(i):
        if (c.iloc[i, j] >= Min_corr_to_delete) and (c.columns[j] not in col_corr):
            colname = c.columns[i] # getting the name of column
            col_corr.add(colname)

cols = X.columns
final_columns = []

for i in cols:
    if i in col_corr:
        continue
    else:
        final_columns.append(i)
                
X1 = X[final_columns]
test1 = test[final_columns]
# ------------ examine the affect of the imbalanced data set ------------#
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2)
lg1 = lgb.LGBMClassifier()
lg1.fit(X_train,y_train)
y_pred = lg1.predict(X_test)
CM = confusion_matrix(y_test, y_pred) 


# ------------ Modeling ------------ #
# ----- ensemble methods ----- #

# --- Light GBM --- #
param_lgb = {"max_depth": [50, 100, 200],
              "learning_rate" : [0.07, 0.05, 0.03],
              "num_leaves": [100],
              "n_estimators": [100, 200, 500],
              "boosting_type": ['gbdt'],
              'min_child_samples': [10, 50, 100],
              'subsample': [0.2, 0.4, 0.6],
              'colsample_bytree': [0.3, 0.5, 0.7],
              'reg_alpha': [0]
             }
lg = lgb.LGBMClassifier()
RSlg = RandomizedSearchCV(lg, param_distributions=param_lgb, n_iter=20, scoring="roc_auc", cv=5, refit=True, verbose=51)
RSlg.fit(X1, y)

# --- tuning around the chosen parameters ---#
param_lgb1 = {"max_depth": [10, 20, 50],
              "learning_rate": [0.001, 0.01, 0.03],
              "num_leaves": [100],
              "n_estimators": [500, 1000, 2000],
              "boosting_type": ['gbdt'],
              'min_child_samples': [50],
              'subsample': [0.4],
              'colsample_bytree': [0.1, 0.3],
              'reg_alpha': [0]
             }
lg = lgb.LGBMClassifier()
RSlg1 = RandomizedSearchCV(lg, param_distributions=param_lgb1, n_iter=15, scoring="roc_auc", cv=5, refit=True, verbose=51)
RSlg1.fit(X1, y)

Public_score_lgb = 0.9233

# --- catboost --- #
param_cat = {'loss_function': ['Logloss', 'CrossEntropy'],
        'depth':[3, 6, 9],
          'iterations': [500, 750, 1000],
          'learning_rate': [0.01, 0.05, 0.1],
          'l2_leaf_reg':[1, 3, 5, 10]}

CB = CatBoostClassifier()
RScb = RandomizedSearchCV(CB, param_distributions=param_cat, n_iter=15, scoring="roc_auc", cv=4, refit=True, verbose=51)
RScb.fit(X1,y)

# --- tuning around the chosen parameters ---#
param_cat1 = {'loss_function': ['CrossEntropy'],
        'depth': [9, 10],
          'iterations': [1200],
          'learning_rate': [0.1, 0.15],
          'l2_leaf_reg': [10, 12],
          'n_estimators': [500, 1000, 2000]
          }

CB = CatBoostClassifier()
RScb1 = RandomizedSearchCV(CB, param_distributions=param_cat1, n_iter=8, scoring="roc_auc", cv=3, refit=True, verbose=51)
RScb1.fit(X1, y, cat_features=list(cat_cols))

# --- xgboost --- #
param_xgb = {'n_estimators': [100, 200, 500],
              'learning_rate': [0.07, 0.05, 0.03],
              'subsample': [0.2, 0.4, 0.6],
              'max_depth': [50, 100, 200],
              'colsample_bytree': [0.3, 0.5, 0.7]
             }

XGB = xgb.XGBClassifier()
RSxgb = RandomizedSearchCV(XGB, param_distributions=param_xgb, n_iter=20, scoring="roc_auc", cv=5, refit=True, verbose=51)
RSxgb.fit(X1, y)


# ------------ predict & submit ------------ #
predictions = RScb.predict_proba(test1)
predictions1 = predictions[:, 1]

sub['isFraud'] = predictions1
sub.to_csv(Working_path+'\\cat.csv', index=False)


# ----- NN ----- #
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})
get_custom_objects().update({'focal_loss_fn': focal_loss()}) 
    
def create_model(loss_fn,firsL_size,activation,first_Drop,secondL_size,second_Drop):
    inps = Input(shape=(X1.shape[1],))
    x = Dense(firsL_size, activation = activation)(inps)
    x = BatchNormalization()(x)
    x = Dropout(first_Drop)(x)
    x = Dense(secondL_size, activation = activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(second_Drop)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(
        optimizer=Adam(),
        loss=[loss_fn]
    )
    #model.summary()
    return model            

def Scale_train_test(train,test):
    cat_train = train[cat_data.columns]
    num_train = train[[col for col in X1.columns if col not in cat_data.columns]]
    cat_test = test[cat_data.columns]
    num_test = test[[col for col in X1.columns if col not in cat_data.columns]]
    stsc = StandardScaler()
    num_train_scald = pd.DataFrame(stsc.fit_transform(num_train))
    num_test_scaled = pd.DataFrame(stsc.transform(num_test))
    return pd.concat([cat_train,num_train_scald],axis = 1),pd.concat([cat_test,num_test_scaled],axis = 1)

X1_scaled,test1_scaled = Scale_train_test(X1,test1)

# --- Use NN as a feature extractor --- #
X_tr, X_val, y_tr, y_val = train_test_split(X1_scaled, y, test_size=0.2)
model_focal = create_model('focal_loss_fn')
model_focal.fit(X_tr, y_tr, epochs=8, batch_size=2048, validation_data=(X_val, y_val), verbose=True)

get_1st_layer_output = K.function([model_focal.layers[0].input], [model_focal.layers[6].output])

X_tr_extractor = get_1st_layer_output([X1_scaled])[0]
test_extractor = get_1st_layer_output([test1_scaled])[0]
     
lg = lgb.LGBMClassifier()
lg.fit(X_tr_extractor, y)
pred_lgb_NN = lg.predict_proba(test_extractor)
pred_lgb_NN = pred_lgb_NN[:, 1]
sub['isFraud'] = pred_lgb_NN
sub.to_csv(Working_path+'\\NN.csv', index=False)

# --- NN tuning --- #
NN = KerasClassifier(build_fn=create_model, verbose=51)
param_NN = {
     'loss_fn': ['focal_loss_fn','binary_crossentropy'],
     'firsL_size': [128, 256, 512, 1024],
    'activation': ['sigmoid', 'relu', 'custom_gelu'],
    'first_Drop': [0.1, 0.3, 0.5],
    'secondL_size': [64, 128, 256],
    'second_Drop': [0.2]
    }

RSnn = RandomizedSearchCV(NN, param_distributions=param_NN, n_iter=15, scoring="roc_auc", cv=3, refit=True, verbose=51)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
RSnn.fit(X1_scaled, y, validation_split=0.20, epochs=16, batch_size=2048, callbacks=[es])
pred_NN = RSnn.predict_proba(X1_scaled)
pred_NN = pred_NN[:, 1]
sub['isFraud'] = pred_NN
sub.to_csv(Working_path+'\\NN_tuned.csv', index=False)

