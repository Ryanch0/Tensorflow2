import tensorflow as tf

# 악보 음표데이터
text = open('musicData.txt','r').read()

# 문자를 숫자로 만들어야함 -> Bag of words(단어주머니)만들기
uniqText = list(set(text)) #중복 허용하지않는 list자료로 만들기
uniqText.sort() #정렬
# print(uniqText)

# utilities
text_to_num = {} # 문자를 숫자로만들기
num_to_text = {} # 숫자를 문자로만들기
for i, data in enumerate(uniqText):
    text_to_num[data] = i
    num_to_text[i] = data
# print(text_to_num)
# print(num_to_text)

# 원본데이터를 최종적으로 숫자화(list자료)
numText = []
for i in text:
    numText.append(text_to_num[i])
# print(numText)

# X,Y데이터셋
X = []
Y = []

#    이 전체가 반복문돌면서 sequence data training 가능
for i in range(0, len(numText) -25 ):
   X.append(numText[i : i + 25]) #i = 0일때 0~24까지를 X데이터로
   Y.append(numText[i + 25]) #i = 0일때 25데이터를 Y데이터로

# print(X[0 : 5])
# print(Y[0  :5])

# 원핫인코딩 왜 31이냐? 유니크문자의 갯수가 0부터 30까지 총 31개였음
X = tf.one_hot(X, 31)
Y = tf.one_hot(Y, 31)


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25,31)),
    tf.keras.layers.Dense(31, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# LSTM은 epochs가 많이필요 100회이상..
model.fit(X,Y, batch_size=64, epochs=100)

model.save('music/models')