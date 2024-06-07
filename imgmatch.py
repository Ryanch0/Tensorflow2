
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# print(trainX.shape)

# 0~255를 0~1로 압축(선택사항)
trainX = trainX / 255.0
testX = testX / 255.0

# 전처리하기
# 원래 데이터는 (60000,28,28) 이었는데 쉐입 바꾸기 -> (60000,28,28,1)전체 데이터에 괄호쳐달라는의미
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

model = tf.keras.Sequential([
    # 컨볼루션 레이어 Conv2D(n개의 이미지 복사본, (n,n 커널 사이즈),etc, activation='relu',input_shape)
    # 이미지엔 음수가 없기에 relu 자주사용
    tf.keras.layers.Conv2D(32,(3,3), padding="same", activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3), padding="same", activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),

    # 행렬을 1차원으로 압축
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    # 마지막 레이어 노드수를 카테고리 갯수만큼해서 확률예측하기
    tf.keras.layers.Dense(10, activation='softmax'),
])

# checkpoint w값만 저장
콜백함수 = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkPoint/mnist',
    monitor='val_acc',
    mode='max',
    save_weights_only=True,
    save_freq='epoch'
)


# summary 보려면 input_shape 넣어야함
model.summary()

optimizier = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001)
model.compile(loss = "sparse_categorical_crossentropy", optimizer=optimizier, metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX,testY) ,epochs=5, callbacks=[콜백함수])

# 학습 후 모델평가(학습용데이터가 아닌)
score = model.evaluate(testX,testY)
print(score)


# 모델 전체저장 및 불러오기
# model.save('newpolder/model1')
# 불러온모델 = tf.keras.models.load_model('newpolder/model1')
# 불러온모델.summary()
# 불러온모델.evaluate(testX,testY)