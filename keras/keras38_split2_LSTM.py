import numpy as np
import numpy as np
from tensorboard import summary
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, GRU
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import EarlyStopping

a = np. array(range(1,101))
x_predict = np.array(range(96, 106))

size = 5 # x는 4개, y는 1개

def split_x(dataset, size): # 시계열 데이터 자르는 함수
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset) # 하나하나 리스트 형태로 추가함
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) # (96, 5)
ccc = split_x(x_predict, 4) # x_predict를 예측할 수 있게 즉 비교할 수 있게 잘라서 어레이로 만들어줘야함
print(ccc.shape) # (7, 4)

x =  bbb[:, :-1]
y =  bbb[:, -1]
print(x, y)
print(x.shape, y.shape) # (96, 4) (96,)

x = x.reshape(96, 4, 1)

# 모델 구성 및 평가 예측할 것

# 2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=False, input_shape=(4,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1)) # Dense는 3차원도 받을 수 있다!
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5000, restore_best_weights=True)
model.fit(x,y, epochs=1000, callbacks=[Es], validation_split=0.1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = ccc.reshape(7, 4, 1)
result = model.predict(x_predict)
print('loss: ', loss)
print('결과: ', result)

# loss:  0.002475091489031911
# 결과:  [[100.24604 ]
#  [101.27447 ]
#  [102.308044]
#  [103.34193 ]
#  [104.37646 ]
#  [105.4114  ]
#  [106.44668 ]]