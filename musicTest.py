import tensorflow as tf
import numpy as np

Pmodel = tf.keras.models.load_model('./model1')

# 테스트용 임의 코드
# chords = [
#     "G", "B", "C", "Cm", "D", "F#m", "Em", "A", "Am", "F",
#     "G7", "B7", "D5", "E7", "D7", "D7#9", "D7b9", "Bb7", "Eb7", "F7",
#     "Gm", "Ab", "E", "A7", "C#m"
# ]
# # 텍스트 파일로 저장
# with open('chords.txt', 'w') as f:
#     for chord in chords:
#         f.write(chord + '\n')


# 악보 음표데이터
text = open('chords.txt','r').read()

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
# print(num_to_text)       #test용 데이터의 코드 갯수는 14개로 이루어짐


numTest = [text_to_num[char] for char in text if char in text_to_num]

# 14개로 이루어진 코드를 모델의 shape에 맞게 (25,31)로 만드는 작업
# 입력 시퀀스 길이 설정
seq_length = 14
target_seq_length = 25  # 모델이 기대하는 시퀀스 길이
num_classes = 31  # 모델이 기대하는 원-핫 인코딩 차원

# 테스트 데이터의 원-핫 인코딩 및 변환
def prepare_input_data(numTest, target_seq_length=25, num_classes=31):
    # 원-핫 인코딩
    one_hot_test = tf.one_hot(numTest, num_classes)
    
    # 필요한 길이만큼 패딩 추가
    if len(one_hot_test) < target_seq_length:
        padding = tf.zeros((target_seq_length - len(one_hot_test), num_classes))
        one_hot_test = tf.concat([one_hot_test, padding], axis=0)
    else:
        one_hot_test = one_hot_test[:target_seq_length]

    # 배치 차원 추가
    one_hot_test = tf.expand_dims(one_hot_test, axis=0)
    return one_hot_test

# 첫 입력 데이터 준비
첫입력 = prepare_input_data(numTest, target_seq_length, num_classes)



music = []
for i in range(200):
    예측값 = Pmodel.predict(첫입력)
    예측값 = np.argmax(예측값[0]) #확률 최댓값 뽑기

    # new예측값 = np.random.choice(uniqText, 1, p=예측값[0]) #루프에 빠졌을경우

    # print(예측값)

    # print(num_to_text[예측값])

    #정답 확인
    # print(numText[117+25])

    # 예측 값을 0부터 13 사이의 값으로 변환 (이건 테스트 데이터의 갯수에따라 달라짐)
    예측값 = 예측값 % 14


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