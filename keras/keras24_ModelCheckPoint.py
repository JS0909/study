from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')
log = model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[es, mcp], validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2:', r2)
