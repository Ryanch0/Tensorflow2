
import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# # print(trainX.shape)

# # 0~255를 0~1로 압축(선택사항)
trainX = trainX / 255.0
testX = testX / 255.0

# # 전처리하기
# # 원래 데이터는 (60000,28,28) 이었는데 쉐입 바꾸기 -> (60000,28,28,1)전체 데이터에 괄호쳐달라는의미
trainX = trainX.resize( 150, 150)
testX = testX.resize(150, 150)


def convert_to_rgb(images):
    return tf.image.grayscale_to_rgb(images)

# Convert the training and test datasets to RGB
trainX = convert_to_rgb(trainX)
testX = convert_to_rgb(testX)

train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY)).map(lambda x, y: (convert_to_rgb(x), y)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY)).map(lambda x, y: (convert_to_rgb(x), y)).batch(32)


model.summary()
history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)


# # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

# # model = tf.keras.Sequential([

# #     # 사진뒤집기 같이 수정해서 이미지 증강시키기
# #     tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(28,28,1)),
# #     tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
# #     tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

# #     # 컨볼루션 레이어 Conv2D(n개의 이미지 복사본, (n,n 커널 사이즈),etc, activation='relu',input_shape)
# #     # 이미지엔 음수가 없기에 relu 자주사용
# #     tf.keras.layers.Conv2D(32,(3,3), padding="same", activation='relu'),
# #     tf.keras.layers.MaxPooling2D((2,2)),
# #     tf.keras.layers.Conv2D(64,(3,3), padding="same", activation='relu'),
# #     tf.keras.layers.MaxPooling2D((2,2)),

# #     # tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),

# #     # 행렬을 1차원으로 압축
# #     tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(64, activation='relu'),
# #     # 마지막 레이어 노드수를 카테고리 갯수만큼해서 확률예측하기
# #     tf.keras.layers.Dense(10, activation='softmax'),
# # ])

# # checkpoint w값만 저장
# # 콜백함수 = tf.keras.callbacks.ModelCheckpoint(
# #     filepath='checkPoint/mnist',
# #     monitor='val_acc',
# #     mode='max',
# #     save_weights_only=True,
# #     save_freq='epoch'
# # )


# # summary 보려면 input_shape 넣어야함
# model.summary()

# optimizier = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001)
# model.compile(loss = "sparse_categorical_crossentropy", optimizer=optimizier, metrics=['accuracy'])

# # TensorBoard 사용법
# from tensorflow.keras.callbacks import TensorBoard
# import time

# # tensorboard = TensorBoard(log_dir='logs/{}'.format('첫모델' +str(int(time.time()))))


# # EarlyStopping
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'val_accuracy', patience=3, mode='max')

# model.fit(trainX, trainY, validation_data=(testX,testY) ,epochs=10, callbacks=[tensorboard, es])

# # 학습 후 모델평가(학습용데이터가 아닌)
# score = model.evaluate(testX,testY)
# print(score)

from tensorflow.keras.applications.inception_v3 import InceptionV3

inception_model = InceptionV3(input_shape=(150,150,3), include_top=False, weights=None)
inception_model.load_weights('inception_v3.h5')

# inception_model.summary()

# w값 업데이트하지않고 고정시키기 (남의거 베껴오는게 목적이니까)
for i in inception_model.layers : 
  i.trainable = False

# 원하는레이어만 뽑기
마지막레이어 = inception_model.get_layer('mixed7')

print(마지막레이어)

layer1 = tf.keras.layers.Flatten()(마지막레이어.output)
layer2 = tf.keras.layers.Dense(1024, activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
output = tf.keras.layers.Dense(10, activation='softmax')(drop1)

model = tf.keras.Model(inception_model.input,output)

model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX,testY) ,epochs=2)


