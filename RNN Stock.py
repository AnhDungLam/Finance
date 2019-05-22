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

"""

#######ANN
X_ann = data.iloc[:, 4:-1]
y_ann = data.iloc[:, -1]
split = int(len(data)*0.8)
X_train_ann, X_test_ann, y_train_ann, y_test_ann =X_ann[:split], X_ann[split:], y_ann[:split], y_ann[split:]

##Scale
from sklearn.preprocessing import StandardScaler
sc_ann = StandardScaler()
X_train_ann = sc_ann.fit_transform(X_train_ann)
X_test_ann = sc_ann.transform(X_test_ann)


ann = Sequential()
ann.add(Dense(units=50, kernel_initializer='uniform', activation='relu', input_dim= X_test_ann.shape[1]))
ann.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
ann.add(Dense(units=50, kernel_initializer='uniform', activation='sigmoid'))
ann.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
ann.fit(X_train_ann,y_train_ann, epochs=10, batch_size=10, verbose=2)


"""

"""
#read file
df = pd.read_csv('NSE-TATAGLOBAL11.csv')
df.head()

data = df.sort_index(ascending = True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date','Close'])
for i in range(0, len(new_data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace =True)

##Creating dataset train and test
dataset = new_data.values
X = pd.DataFrame(dataset)
limit_train =int(len(dataset)*training_Set_percent)-1
train = dataset[0:limit_train, :]
valid = dataset[limit_train:, :]

#Scale dataset train
scale = MinMaxScaler(feature_range=(0,1))
scaled_data = scale.fit_transform(dataset)

x_train, y_train =[],[]

for i in range(varTimeStamp, len(train)):
    x_train.append(scaled_data[i-varTimeStamp:i,0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

####model rnn
rnn = Sequential()
rnn.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1) ))
rnn.add(LSTM(units=50))
rnn.add(Dense(1))

rnn.compile(optimizer='adam', loss='mean_squared_error')
rnn.fit(x_train,y_train,epochs=10, batch_size=1, verbose=2)

####Predict model
inputs = new_data[len(new_data)-len(valid)-varTimeStamp:].values
inputs = inputs.reshape(-1,1)
inputs = scale.transform(inputs)

x_test = []
for i in range(varTimeStamp, inputs.shape[0]):
    x_test.append(inputs[i-varTimeStamp:i,0])
x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
closing_price = rnn.predict(x_test)
closing_price = scale.inverse_transform(closing_price)

train = new_data[:limit_train]
valid = new_data[limit_train:]
valid['Pred'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Pred']])
"""