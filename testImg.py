import tensorflow as tf
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

model2 = tf.keras.Sequential([
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

model2.summary()

optimizier = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001)
model2.compile(loss = "sparse_categorical_crossentropy", optimizer=optimizier, metrics=['accuracy'])

# checkpoint에서 w값 불러오기
model2.load_weights('checkPoint/mnist')

model2.evaluate(testX,testY)