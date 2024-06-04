import pandas as pd
import tensorflow as tf
import numpy as np

# padas라이브러리로 csv파일 가져오기 변수에 저장
data = pd.read_csv('gpascore.csv')

# 데이터에 비어있는 값의 개수
# print(data.isnull().sum())
# 빈데이터 없애기함수
data = data.dropna()

# 해당 열의 최솟값
# print(data['gpa'].min())

# # 빈칸 없애기(값)
# data.fillna(100)

# 해당 열 데이터 리스트자료로 만들어줌
y데이터 = data['admit'].values

x데이터 = []

# pandas 반복문 하나의 열로 추출
for i, rows in data.iterrows():
   x데이터.append([rows['gre'],rows['gpa'],rows['rank']])

print(x데이터)
    
model = tf.keras.models.Sequential([
    # 레이어만들기(노드 수,활성함수)
    # 중간레이어
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # 최종레이어(sigmoid는 모든값 0~1로 만들기때문에 최종 확률로 해석 가능)
    tf.keras.layers.Dense(1,activation='sigmoid'),
])

# 컴파일까지 해줘야함, bianary_crossentropy로스함수는 0~1값,확률문제에서 적절
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습시키기 (x데이터는 학습데이터, y데이터는 정답데이터, epochs=학습횟수)
model.fit(np.array(x데이터),np.array(y데이터), epochs=1000 )

# 예측
예측값 = model.predict([[800,4.0,4],[800,4.0,1]])
print(예측값)