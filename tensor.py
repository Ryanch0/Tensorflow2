import tensorflow as tf

tensor = tf.constant([3,4,5])
tensor2 = tf.constant([3,4,5])
tensor3 = tf.constant([[1,2],
                       [3,4]])
tensor4 = tf.constant([[5,2],
                       [3,4]])
print(tf.matmul(tensor3,tensor4))


tensor6 = tf.zeros([2,2])
print(tensor4.shape)

# weight Variable
w = tf.Variable(1.0) 
print(w)