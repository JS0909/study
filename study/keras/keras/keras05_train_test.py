import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7]) # (7,)
x_test = np.array([8,9,10])         # (3,)
y_train = np.array([1,2,3,4,5,6,7]) # (7,)
y_test = np.array([8,9,10])         # (3,)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 평가 데이터의 loss 
                                        # (데이터 사이즈가 작으므로 통상적으로 훈련 loss보다 나쁘게 나온다)
print("loss : ", loss)
result = model.predict([11])
print('[11]의 예측값 : ', result)
