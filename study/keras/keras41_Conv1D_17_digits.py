from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
# 이미지 파일 8x8의 픽셀 한칸짜리가 1797개가 있음, 각 픽셀에 칠해진 여부에 때라 0 ~ 255의 숫자로 표시
print(np.unique(y, return_counts=True)) # [0 1 2 3 4 5 6 7 8 9] <-이 숫자를 찾아라
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(1437, 64, 1)
x_test = x_test.reshape(360, 64, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(80, 2, input_shape=(64, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)
end_time = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1) # 행끼리 비교해서 몇번째 인덱스가 제일 큰지 알려줌
y_test = tf.argmax(y_test, axis=1) # y_test도 argmax를 해서 같은 리스트를 비교하기

acc_sc = accuracy_score(y_test, y_predict) # 비교
print('acc스코어 : ', acc_sc)
print('걸린 시간: ', end_time-start_time)
print('digits')

# MinMax scaler
# loss :  0.11650712043046951
# acc스코어 :  0.9638888888888889
# 걸린 시간:  3.039471387863159

# LSTM
# loss :  0.5667863488197327
# acc스코어 :  0.7972222222222223
# 걸린 시간:  67.40128946304321

# Conv1D
# loss :  0.16208083927631378
# acc스코어 :  0.9527777777777777
# 걸린 시간:  7.133979082107544