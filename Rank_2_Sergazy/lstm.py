import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import re
import nltk
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


y_train = train.final_status
train = train[['goal','desc', 'name', 'keywords']]
test = test[['goal','desc', 'name', 'keywords']]
# Keeping only the neccessary columns
data = pd.concat([train, test])
data = data.set_index(np.arange(len(data)))
data['desc'] = data['desc'] + data['name'] + data['keywords']

data['desc'] = data['desc'].apply(lambda x: str(x).lower())
data['desc'] = data['desc'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))


max_features = 6000
tokenizer = Tokenizer(nb_words=max_features, split=' ')
tokenizer.fit_on_texts(data['desc'].values)
X = tokenizer.texts_to_sequences(data['desc'].values)
X = pad_sequences(X)
embed_dim = 256
lstm_out = 512
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2, return_sequences=True))
model.add(GRU(lstm_out, activation='relu'))
model.add(Dense(lstm_out, input_dim=lstm_out, activation='tanh'))
model.add(Dense(lstm_out, input_dim=lstm_out, activation='relu'))
model.add(Dense(lstm_out, input_dim=lstm_out, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

y_train = pd.get_dummies(y_train).values
x_train = X[:len(train)]
x_test = X[len(train):]

batch_size = 64
model.fit(x_train, y_train, nb_epoch=2, batch_size=batch_size, verbose=2)
Y = model.predict_proba(X)

np.savetxt('lstm', Y, delimiter=',', fmt = '%0.6f')