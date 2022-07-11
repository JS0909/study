from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential, load_model, Input, Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.layers import BatchNormalization

# 1. 데이터
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path+'test.csv', index_col=0)
print(test_set)
print(test_set.shape) # (180, 11)

print(train_set.info())
print(train_set.isnull().sum())

train_set = train_set.fillna(0)
test_set = test_set.fillna(0)

def get_month(date) : # 월만 빼오기
    month = date[3:5]
    month = int(month)
    return month

train_set['Month'] = train_set['Date'].apply(get_month)
test_set['Month'] = test_set['Date'].apply(get_month)

def holiday_to_number(isholiday):
    if isholiday == True:
        number = 1
    else:
        number = 0
    return number

train_set['NumberHoliday'] = train_set['IsHoliday'].apply(holiday_to_number)
test_set['NumberHoliday'] = test_set['IsHoliday'].apply(holiday_to_number)
train_set = train_set.drop(['IsHoliday', 'Date'], axis=1)
test_set = test_set.drop(['IsHoliday', 'Date'], axis=1)

x = train_set.drop(['Weekly_Sales'], axis=1)
y = train_set['Weekly_Sales']

print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# # 2. 모델 구성
# 시퀀셜
# model = Sequential()
# model.add(Dense(32, input_dim=13))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))

# 함수형
input1 = Input(shape=(11,))
dense1 = Dense(50,activation='relu')(input1)
batchnorm1 = BatchNormalization()(dense1)
dense2 = Dense(100)(batchnorm1)
dense3 = Dense(200)(dense2)
drop1 = Dropout(0.3)(dense3)
dense4 = Dense(200, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense4)
batchnorm2 = BatchNormalization()(drop2)
dense5 = Dense(100)(batchnorm2)
drop3 = Dropout(0.2)(dense5)
dense6 = Dense(50, activation='relu')(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[Es], validation_split=0.25)
# model.save('./study/_save/keras32.msw')

# model = load_model('./study/_save/keras32.msw')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

def RMSE(y_test, y_predict): # rmse 계산 사용 법
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_predict, y_test)
print('rmse: ', rmse)

# 5. 제출 준비
submission = pd.read_csv(path + 'submission.csv', index_col=0)
y_submit = model.predict(test_set)
submission['Weekly_Sales'] = y_submit
submission.to_csv(path + 'submission.csv', index=True)


# loss:  [177036378112.0, 323372.75]
# r2:  0.46274552754920784
# rmse:  420756.8833430339

# loss:  [165164056576.0, 302009.875]
# r2:  0.45754306042863324
# rmse:  406403.8384129067

# loss:  [194598109184.0, 328610.9375]
# r2:  0.39788776981340923
# rmse:  441132.7451355292