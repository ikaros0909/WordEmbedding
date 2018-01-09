'''
Created on 2017. 5. 16.

@author: danny
jupyter notebook --ip=* --no-browser
https://www.lucypark.kr/courses/2015-ba/text-mining.html
'''
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
import time, datetime
start = time.time()
print("시작시간 : %s" % datetime.datetime.now())
######################################################################
# 1.Load data
######################################################################
# Read excel file
def data_from_xlsx():
    # Read excel file
    df = pd.read_excel("TeacherComment.xlsx")
    print("DB출력 : %s" % datetime.datetime.now())
    return df

######################################################################
# Tagging and chunking
######################################################################
docs_ko = data_from_xlsx()

from konlpy.tag import Twitter; 
t = Twitter()

tags_ko = [t.pos(doc) for doc in docs_ko.Content]
#     print("tags_ko : %s" % (tags_ko))

# pos = lambda d: ['/'.join(p) for p in t.nouns(d)]
pos = lambda d: [p for p in t.nouns(d) if len(p)>1]
Nouns_ko = [pos(doc) for doc in docs_ko.Content]
#     print("Nouns_ko : %s" % (Nouns_ko))
print("명사추출 : %s" % datetime.datetime.now())

from gensim.models import word2vec
model_file = 'ko_word2vec_e.model'

import os
if os.path.exists(model_file):
    wv_model_ko = word2vec.KeyedVectors.load(model_file) #저장된 모델 불러오기

#size=100 : 콘텐츠를 100차원의 벡터로 바꾸기 
#window=5 : 주변단어는 window parameter로 지정(자신 포함 앞2, 뒤2)
#min_count=3 : corpus내 출현 빈도가 3 미만인 단어는 분석에서 제외하기. 
#workers=4 : CPU는 쿼드코어를 쓰고 iter parameter로 반복 학습 수 지정
#iter=2 : 반속 학습 수
#sg=1 : 분석 방법론은 CBOW = 0 SKIP-GRAM = 1 로써 sg parameter로 지정
wv_model_ko = word2vec.Word2Vec(Nouns_ko, size=100, window=5, min_count=5, workers=4, iter=30, sg=1)
wv_model_ko.init_sims(replace=True)
wv_model_ko.save(model_file) #모델저장하기
# wv_model_ko = word2vec.Word2Vec.load(model_file) 
print(wv_model_ko.wv.vocab)

#most_similar함수는 두 벡터 사이의 코사인 유사도를 구해주는데, 그 값이 작을 수록 비슷한 단어라는 뜻입니다. 
#most_similar(positive=["디자인"], topn=100) 디자인이라는 단어와 가장 비슷한(코사인 유사도가 큰) 100개 단어를 출력하라는 의미
similar_word = wv_model_ko.most_similar(pos('진심'), topn=10)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("similar_word : %s" % similar_word)

# word_vectors = word2vec.KeyedVectors.load(model_file) #저장된 모델 불러오기
# print("word_vectors : %s" % (wv_model_ko.most_similar(positive=[pos('회장'), pos('이끌며')], negative=[pos('부원'), pos('예비')])))

print("종료시간 : %s" % datetime.datetime.now())
end = time.time() - start
print("실행시간(초) : %s" % end)














