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

X = []
Y = []