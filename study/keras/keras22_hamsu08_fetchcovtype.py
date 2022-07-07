# keras18_gpu_test3 파일의 서머리를 확인해보시오.

from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # [1 2 3 4 5 6 7]
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))

#2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim=54, activation='relu'))
# model.add(Dense(200))
# model.add(Dense(150))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(7, activation='softmax'))

input1 = Input(shape=(54,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(200)(dense1)
dense3 = Dense(150)(dense2)
dense4 = Dense(300, activation='relu')(dense3)
dense5 = Dense(200, activation='relu')(dense4)
output1 = Dense(7, activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=100, batch_size=50, callbacks=[Es], validation_split=0.2)
end_time = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)
print('걸린 시간: ', end_time-start_time)

# 시퀀셜 RobustScaler
# loss :  0.24489960074424744
# acc스코어 :  0.9021023553608771
# 걸린 시간:  102.6204833984375

# 함수 RobustScaler
# loss :  0.23509512841701508
# acc스코어 :  0.9077390428818533
# 걸린 시간:  105.27877497673035