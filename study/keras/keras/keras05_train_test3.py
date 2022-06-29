import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라.
# 힌트: 사이킷런

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=7, random_state=99)

print(x_test)
print(x_train)
print(y_test)
print(y_train)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict([11])
print('[11]의 예측값 : ', result)
