import pandas as pd
import numpy as np
import enchant
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from word2vecUtils import utils
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
import re
import readability
from nltk.sentiment.vader import SentimentIntensityAnalyzer

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# trainp = np.loadtxt('trainp.csv', delimiter=',')
# testp = np.loadtxt('testp.csv', delimiter=',')
# trainp = pd.DataFrame(trainp)
# testp = pd.DataFrame(testp)
# train = pd.concat([train,trainp],axis = 1)
# test = pd.concat([test,testp],axis = 1)

train['created_atX'] = train['created_at']/max(train['created_at'])
test['created_atX'] = test['created_at']/max(test['created_at'])
train['deadlineX'] = train['deadline']/max(train['deadline'])
test['deadlineX'] = test['deadline']/max(test['deadline'])

y_train = train.final_status
X_train = train.drop(['backers_count', 'final_status'], 1)
X_test = test

X = pd.concat([X_train, X_test])

X = X.set_index(np.arange(len(X)))


def computeRead(text):
    rd = readability.Readability(text)
    score = rd.FleschKincaidGradeLevel()
    return int(score)
def ARIscore(text):
    rd = readability.Readability(text)
    score = rd.ARI()
    return float(score)
def LIXscore(text):
    rd = readability.Readability(text)
    score = rd.LIX()
    return float(score)


X['readscore'] = X['desc'].apply(lambda d: computeRead(str(d)))
X['ariscore'] = X['desc'].apply(lambda d: ARIscore(str(d)))
X['lixscore'] = X['desc'].apply(lambda d: LIXscore(str(d)))

X['readscoreX'] = X['name'].apply(lambda d: computeRead(str(d)))
X['ariscoreX'] = X['name'].apply(lambda d: ARIscore(str(d)))
X['lixscoreX'] = X['name'].apply(lambda d: LIXscore(str(d)))

X['coeff'] = np.zeros(len(X))
X.coeff.ix[X.currency == 'USD'] = 1
X.coeff.ix[X.currency == 'GBP'] = 0.78
X.coeff.ix[X.currency == 'EUR'] = 0.89
X.coeff.ix[X.currency == 'CAD'] = 1.32
X.coeff.ix[X.currency == 'AUD'] = 1.31
X.coeff.ix[X.currency == 'SEK'] = 8.71
X.coeff.ix[X.currency == 'NZD'] = 1.38
X.coeff.ix[X.currency == 'DKK'] = 6.63
X.coeff.ix[X.currency == 'NOK'] = 8.42
X.coeff.ix[X.currency == 'CHF'] = 0.97
X.coeff.ix[X.currency == 'MXN'] = 17.95
X.coeff.ix[X.currency == 'SGD'] = 1.38
X.coeff.ix[X.currency == 'HKD'] = 7.8

X['dollars'] = X['goal'] / X['coeff']

X = pd.get_dummies(X, columns=['country'])



le = LabelEncoder()
le.fit(X.disable_communication)
X.disable_communication = le.transform(X.disable_communication)

le = LabelEncoder()
le.fit(X.currency)
X.currency = le.transform(X.currency)


def year(date):
    return int(time.strftime("%Y", time.localtime(date)))


def month(date):
    return int(time.strftime("%m", time.localtime(date)))


X['created_month'] = np.zeros(len(X))
X['deadline_month'] = np.zeros(len(X))
X['launched_month'] = np.zeros(len(X))
X['state_changed_month'] = np.zeros(len(X))

X['created_month'] = X['created_at'].apply(month)
X['deadline_month'] = X['deadline'].apply(month)
X['launched_month'] = X['launched_at'].apply(month)
X['state_changed_month'] = X['state_changed_at'].apply(month)

d = enchant.Dict("en_US")
X['valideng'] = X['desc'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', str(x))))
X['valideng'] = X['valideng'].apply(lambda x: sum(1 for c in str(x).split(' ') if len(c) < 4 or d.check(c)))
analyzer = SentimentIntensityAnalyzer()
def compoundScore(text):
    res = analyzer.polarity_scores(text)
    return float(res['compound'])
def negSent(text):
    res = analyzer.polarity_scores(text)
    return float(res['neg'])
def posSent(text):
    res = analyzer.polarity_scores(text)
    return float(res['pos'])
def neuSent(text):
    res = analyzer.polarity_scores(text)
    return float(res['neu'])
X['compoundScore'] = X['desc'].apply(lambda d: compoundScore(str(d)))
X['negSent'] = X['desc'].apply(lambda d: negSent(str(d)))
X['posSent'] = X['desc'].apply(lambda d: posSent(str(d)))
X['neuSent'] = X['desc'].apply(lambda d: neuSent(str(d)))

X['compoundScoreX'] = X['name'].apply(lambda d: compoundScore(str(d)))
X['negSentX'] = X['name'].apply(lambda d: negSent(str(d)))
X['posSentX'] = X['name'].apply(lambda d: posSent(str(d)))
X['neuSentX'] = X['name'].apply(lambda d: neuSent(str(d)))

cols_to_use = ['name', 'desc']
len_feats = ['name_len', 'desc_len']
count_feats = ['name_count', 'desc_count']

for i in np.arange(2):
    X[len_feats[i]] = X[cols_to_use[i]].apply(str).apply(len)
    X[count_feats[i]] = X[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))

X['keywords_len'] = X['keywords'].apply(str).apply(len)
X['keywords_count'] = X['keywords'].apply(str).apply(lambda x: len(x.split('-')))

X['dots'] = X['desc'].apply(str).apply(lambda x: x.count('.'))
X['comma'] = X['desc'].apply(str).apply(lambda x: x.count(','))
X['kav'] = X['desc'].apply(str).apply(lambda x: x.count('\"'))
X['vopros'] = X['desc'].apply(str).apply(lambda x: x.count('?'))
X['voskl'] = X['desc'].apply(str).apply(lambda x: x.count('!'))
X['smiles'] = X['desc'].apply(str).apply(lambda x: x.count(":)"))
X['Iocc'] = X['desc'].apply(str).apply(lambda x: x.count('I') + x.count('i'))
X['kkstid'] = X['project_id'].apply(str).apply(lambda x: int(x.replace('kkst', '')))

X['digitsenc'] = X['desc'].apply(str).apply(
    lambda x: x.count('0') + x.count('1') + x.count('2') + x.count('3') + x.count('4') + x.count('5') + x.count(
        '6') + x.count('7') + x.count('8') + x.count('9'))

X['kkstidlen'] = X['project_id'].apply(str).apply(len)
X['potentiality'] = (X['deadline'] - X['created_at']) * X['dollars']
X['hardness'] = X['dollars'] / (X['deadline'] - X['created_at'])
X['freshness'] = X['deadline'] / X['state_changed_at']
X['editingTime'] = X['created_at'] / X['launched_at']
X['diversity'] = X['keywords_len'] / X['name_len']
X['diversity2'] = X['desc_count'] / X['keywords_count']
X['upper'] = X['desc'].apply(str).apply(lambda x: sum(1 for c in x if c.isupper()))

X['editingDuration'] = np.log(X['launched_at'] - X['created_at'])
X['loggoal'] = np.log(X['dollars'])
X['durationX'] = np.log(X['deadline'] - X['launched_at'])

#from datetime import datetime
#X['satornot'] = np.zeros(len(X))
#X['dow'] = X['deadline'].apply(lambda x: datetime.fromtimestamp(x/1000).strftime("%A"))
#X.satornot.ix[X.dow == 'Saturday'] = 1
#X.satornot.ix[X.dow != 'Saturday'] = 0
#X = X.drop(['dow'], 1)

#X['durationToChange'] = X['state_changed_at'] - X['deadline']
import datetime
daydict = {}
for index, row in X.iterrows():
    if datetime.datetime.fromtimestamp(int(row['deadline'])).strftime('%Y-%m-%d') in daydict:
        daydict[datetime.datetime.fromtimestamp(int(row['deadline'])).strftime('%Y-%m-%d')] += 1
    else:
        daydict[datetime.datetime.fromtimestamp(int(row['deadline'])).strftime('%Y-%m-%d')] = 0

X['zagr'] = X['deadline'].apply(lambda x: daydict[datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d')])
# time.strftime("%Y", time.localtime(X.deadline))
# time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(epoch))


# clean_desc= []
# for index,row in X.iterrows():
#     clean_desc.append(" ".join(utils.review_to_wordlist(str(row['desc']) + "" + str(row['name']) + " " + str(row['keywords']), False)))
#
# vectorizer2  =  TfidfVectorizer(min_df=3,  max_features=300,
#         strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#         ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
#         stop_words = None )
#
# data_features = vectorizer2.fit_transform(clean_desc)
# np.asarray(data_features)
# data_features = data_features.astype(np.float64)
# features_df = pd.DataFrame(data_features.todense(), columns=vectorizer2.get_feature_names())
# X = pd.concat([X, features_df], axis=1)

clean_desc = []
for index, row in X.iterrows():
    clean_desc.append(" ".join(
        utils.review_to_wordlist(str(row['desc']) + "" + str(row['name']) + " " + str(row['keywords']), False)))

vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 3300)

data_features = vectorizer.fit_transform(clean_desc)
np.asarray(data_features)
data_features = data_features.astype(np.float32)
features_df = pd.DataFrame(data_features.todense(), columns=vectorizer.get_feature_names())
X = pd.concat([X, features_df], axis=1)

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=15, random_state=0).fit(features_df)
# labels = kmeans.labels_
# kmeans = []
# X['categoryX'] = labels
#
# X = pd.get_dummies(X, columns=['categoryX'])


#
# def tokenizerKeras(data):
#     data = data[['desc']]
#
#     data['desc'] = data['desc'].apply(lambda x: str(x).lower())
#     data['desc'] = data['desc'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
#
#     for idx, row in data.iterrows():
#         row[0] = row[0].replace('rt', ' ')
#
#     max_features = 1500
#     tokenizer = Tokenizer(nb_words=max_features, split=' ')
#     tokenizer.fit_on_texts(data['desc'].values)
#     X = tokenizer.texts_to_sequences(data['desc'].values)
#     X = pad_sequences(X)
#     return X
# features_df = pd.DataFrame(tokenizerKeras(X))
# X = pd.concat([X, features_df], axis=1)

X = X.drop(['project_id', 'name', 'desc', 'keywords'], 1)
# colnames = list(X.columns.values)
# todrop = []
# for col in colnames:
#     try:
#         cur = col.astype(int)
#         todrop.append(col)
#     except:
#         continue
# X.drop(todrop, 1)

# cols = X.columns
# for dup in X.columns:
#     cols[X.columns.get_loc(dup)] = [dup + '.' + str(d_idx) if d_idx != 0 else dup for d_idx in
#                                     range(X.columns.get_loc(dup).sum())]
# X.columns = cols
X_train = X.ix[:len(X_train) - 1]
X_test = X.ix[len(X_train):]
print("started training")

gbm = lgb.LGBMClassifier(n_estimators=2900, max_depth=3, subsample=0.7, colsample_bytree= 0.7)
gbm = gbm.fit(X_train, y_train)
Y = gbm.predict_proba(X)
np.savetxt('lgb',Y,delimiter = ',', fmt = '%0.6f')


