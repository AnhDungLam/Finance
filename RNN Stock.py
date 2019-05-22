# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:37:52 2019

@author: Andy LAM
"""

import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Configuration
varTimeStamp = 60
training_Set_percent = 0.8

##data = pd.read_csv("NSE-TATAGLOBAL11.csv")
data = pd.read_csv("NSE-TATAGLOBAL.csv")

dataset= pd.DataFrame(index=range(0,len(data)), columns=['Date','Close'])
for i in range(0, len(data)):
    dataset['Date'][i] = data['Date'][i]
    dataset['Close'][i] = data['Close'][i]
dataset.index = dataset['Date']
dataset.sort_index(ascending=True, axis=0)
dataset.drop('Date', axis=1,inplace=True)

##Create dataset train and test
train, test  = [],[]
limit_train = int(len(dataset) * training_Set_percent)-1
train = dataset[:limit_train]
valid = dataset[limit_train:]

sc = MinMaxScaler(feature_range=(0,1))
scaled_dataset= sc.fit_transform(dataset)

X_train, y_train =[], []
for i in range(varTimeStamp, len(train)):
    X_train.append(scaled_dataset[i-varTimeStamp:i,0])
    y_train.append( scaled_dataset[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train_RNN = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


##RNN model
rnn = Sequential()
rnn.add(LSTM(units=50, return_sequences=True, input_shape = (X_train_RNN.shape[1],1)))
#rnn.add(LSTM(units=50,return_sequences=True))
#rnn.add(LSTM(units=50,return_sequences=True))
rnn.add(LSTM(units=50))
rnn.add(Dense(1))

rnn.compile(optimizer='adam', loss='mean_squared_error')
rnn.fit(X_train_RNN,y_train, epochs=10,verbose=2, batch_size=1)

##Predict
testset= dataset[len(dataset)-len(valid)-varTimeStamp:].values
testset = sc.transform(testset)
X_test = []
for i in range(varTimeStamp, len(testset)):
    X_test.append(testset[i-varTimeStamp:i,0])
X_test = np.array(X_test)
X_test_RNN = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

_pred = rnn.predict(X_test_RNN)
pred = sc.inverse_transform(_pred) 

###Get score


###Plot
valid['Pred'] = pred

plt.plot(train['Close'])
plt.plot(valid[['Close','Pred']])


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(valid['Close'], valid['Pred']))
print("MSE: %.1f" %rmse)
