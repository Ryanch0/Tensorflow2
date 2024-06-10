import tensorflow as tf
import numpy as np

Pmodel = tf.keras.models.load_model('./model1')

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

# 임의로 뽑기
첫입력 = numText[117 : 117+25]
# 전처리
첫입력 = tf.one_hot(첫입력, 31)
첫입력 = tf.expand_dims(첫입력, axis=0) # 3차원으로 만들기
# print(첫입력)


music = []
for i in range(200):
    예측값 = Pmodel.predict(첫입력)
    예측값 = np.argmax(예측값[0]) #확률 최댓값 뽑기

    # new예측값 = np.random.choice(uniqText, 1, p=예측값[0]) #루프에 빠졌을경우

    # print(예측값)

    # print(num_to_text[예측값])

    #정답 확인
    # print(numText[117+25])

    # 예측 저장
    music.append(예측값)

    다음입력 = 첫입력.numpy()[0][1:] #tensor로 바꾸기, 0번째항목 자르기
    # print(다음입력)

    one_hot_num = tf.one_hot(예측값, 31)
    # print(one_hot_num) #one_hot인코딩한거

    첫입력 = np.vstack([다음입력, one_hot_num.numpy()])
    첫입력 = tf.expand_dims(첫입력, axis=0)

# print(music)

music_text = []

for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))