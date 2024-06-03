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

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)


def 손실함수(a,b):
    예측_y = train_x * a +b
    return tf.keras.losses.mse(train_y, 예측_y)

opt = tf.keras.optimizers.legacy.Adam(learning_rate= 0.1)

for i in range(1000):
    opt.minimize(lambda:손실함수(a,b), var_list=[a,b])
    print(a.numpy(),b.numpy())
