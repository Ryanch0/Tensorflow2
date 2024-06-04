
import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# print(trainX.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    # 행렬을 1차원으로 압축
    tf.keras.layers.Flatten(),
    # 마지막 레이어 노드수를 카테고리 갯수만큼해서 확률예측하기
    tf.keras.layers.Dense(10, activation='softmax'),
])

# summary 보려면 input_shape 넣어야함
model.summary()


model.compile(loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(trainX, trainY, epochs=5)
