from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(16512, 4, 2)
x_test = x_test.reshape(4128, 4, 2)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(5, 2, input_shape=(4, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(30))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
start_time=time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=50,
                callbacks=[earlyStopping],
                validation_split=0.25)
end_time=time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
print('시간: ', end_time-start_time)

# loss :  [0.5216111540794373, 0.546445369720459]
# r2스코어 :  0.625834467060862

# LSTM
# loss :  [0.290325403213501, 0.36063283681869507]
# r2스코어 :  0.7884188322471288

# Conv1D
# loss :  [0.2953258156776428, 0.37131211161613464]
# r2스코어 :  0.7881549313302594
# 시간:  171.27732038497925