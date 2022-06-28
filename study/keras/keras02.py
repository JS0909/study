# 1. 데이터
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np  
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성 
model = Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))
model.add(Dense(9))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('[6]의 예측값 : ', result)

# 결과치 : 5.7279534