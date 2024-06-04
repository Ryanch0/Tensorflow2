import tensorflow as tf

# height = 170
# shoes = 260
# # shoes = height*a + b

# a = tf.Variable(0.1)
# b = tf.Variable(0.2)

# def 손실함수():
#     예측값 = height*a + b
#     return tf.square(260 - 예측값 )
# # 경사하강법
# opt = tf.keras.optimizers.legacy.Adam(learning_rate= 0.01)

# for i in range(300):
#     opt.minimize(손실함수, var_list=[a,b])
#     print(a.numpy(),b.numpy())

# train_x = [1,2,3,4,5,6,7]
# train_y = [3,5,7,9,11,13,15]

# a = tf.Variable(0.1)
# b = tf.Variable(0.1)


# def 손실함수(a,b):
#     예측_y = train_x * a +b
#     return tf.keras.losses.mse(train_y, 예측_y)

# opt = tf.keras.optimizers.legacy.Adam(learning_rate= 0.1)

# for i in range(1000):
#     opt.minimize(lambda:손실함수(a,b), var_list=[a,b])
#     print(a.numpy(),b.numpy())


육모 = [50, 60, 70, 80]
구모 = [60, 64, 77, 90]

수능 = [70, 80, 85, 94]

# 육모*w1 + 구모*w2 = 수능

w1 = tf.Variable(0.1)
w2 = tf.Variable(0.1)

def lossFunc(w1,w2):
    예측 = tf.sigmoid(육모*w1 + 구모*w2)
    return tf.keras.losses.mse(수능, 예측)

opt = tf.keras.optimizers.legacy.Adam(learning_rate= 0.001)

for i in range(3000):
    opt.minimize(lambda:lossFunc(w1,w2), var_list=[w1,w2])
    print(w1.numpy(),w2.numpy())

예측 = tf.sigmoid(육모 * w1 + 구모 * w2)
오차 = 수능 - 예측.numpy()

print(f'Predictions: {예측.numpy()}')
print(f'Errors: {오차}')
