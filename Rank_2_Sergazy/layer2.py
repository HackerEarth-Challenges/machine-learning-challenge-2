import pandas as pd
import numpy as np
import lightgbm as lgb
X1 = pd.read_csv('lgb', delimiter = ',', header = None)
X2 = pd.read_csv('lstm', delimiter = ',', header = None)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train.final_status
X = pd.concat([X1,X2], 1)
X_train = X.ix[:len(train) - 1]
X_test = X.ix[len(train):]
gbm  = lgb.LGBMClassifier()
gbm.fit(X_train,y_train)
y_pred = gbm.predict_proba(X_test)
y_result = []
magic = 0.64
for i in range(0, 63465):
    if y_pred[i][0] > magic:
        y_result.append(0)
    else:
        y_result.append(1)
for index, row in test.iterrows():
    if str(row['name']).count("Canceled") + str(row['name']).count("Suspended") > 0 or row['deadline'] > row['state_changed_at'] or row['disable_communication'] == True:
        y_result[index] = 0
sub = pd.read_csv('samplesubmission.csv')
sub.final_status = y_result
sub.to_csv('sub2.csv', index=0)