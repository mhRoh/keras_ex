import numpy
from keras.models import Sequential
from keras.layers import Dense
# 케라스 북에서는 아래의 부분을 입력 노드 1개 완전 연결 노드 1개 출력 노드 1개로 정의를 하였는데
# 그렇다면 x 가 입력 이며, 2 * x + 1 이 완전 연결 노드 그리고 y가  출력 노드 1개 인듯 하다.

x = numpy.array([0,1,2,3,4])
y = 2 * x + 1

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile("SGD", "mse")

model.fit(x[:3], y[:3], epochs=1000, verbose=0)

print("Predictions : ", model.predict(x[2:]).flatten())

