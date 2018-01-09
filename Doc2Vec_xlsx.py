'''
Created on 2017. 11. 17.

@author: danny
'''
#-*- coding: utf-8 -*-
import gensim
import collections
import smart_open
import random
import re
import os
import pandas as pd
import sys
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import KeyedVectors
'''
import openpyxl
excel_document = openpyxl.load_workbook("teacher_comment.xlsx")
print(type(excel_document))
'''
def data_from_xlsx():
    # Read excel file
    df = pd.read_excel("TeacherComment.xlsx")
    return df


# Define a Function to Read and Preprocess Text
def read_corpus(data, tokens_only=False):
        for i, line in enumerate(data):        
            if tokens_only:
                yield line
            else:
                yield gensim.models.doc2vec.TaggedDocument(line.split(), [i])


matrix = []
test_matrix=[]
data = data_from_xlsx()
for i in data['Content'].tolist() :
    matrix.append(i)
    test_matrix.append(i)

train_corpus = list(read_corpus(matrix)) 
test_corpus = list(read_corpus(test_matrix))

def train():
    # Train doc2vec
    # Build a Vocabulary
    # dm=1 이면 distributed memory 쓰는 것, 그게 아니면 distributed bag of words 쓰는 것 
    # size는 feature 벡터의 차원 
    # negative 가 0 초과면 negative sampling을 사용하면서 얼마나 많은 noise words를 사용할 것인지 5-20까지의 
    # dm_mean 이 0이면 (default), word vector들의 문백을 더하는 걸 사용한다. 1이라면 중앙값?평균?을 사용함. 이건 dm이 non-concatenative 모드에서 사용될 때만 적용된다. 
    # dbow_words 1이면 word-vectors를 train... 
    model = gensim.models.doc2vec.Doc2Vec(dm=0, size=300, min_count=2, iter=30, alpha=0.025, negative=20, dbow_words=1, workers=8, window=10)
    model.build_vocab(train_corpus)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train(train_corpus, total_examples=model.corpus_count, epochs = model.iter)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
        print ("epoch %i complete." %epoch)

    # save model ,, doc2vec and word2vec    
    model.init_sims(replace=True)
    model.save('not_preprocess.doc2vec')
    word2vec_file = "not_preprocess"+".word2vec_format"
    model.save_word2vec_format(word2vec_file, binary = False)

    # Assessing Model
    ranks = []
    first_ranks=[]
    second_ranks = []
    model.init_sims(replace=True)

    #바뀐 부분은 infer vector 에서 most_similar로 !! 
    memory = []
    for doc_id in range(len(train_corpus)):
        sims = model.docvecs.most_similar([model.docvecs[doc_id]], topn=len(model.docvecs))   
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        first_ranks.append(sims[0])
        second_ranks.append(sims[1])    
        #print(model.docvecs[doc_id]) # [숫자, 숫자, 숫자.....][숫자, 숫자, 숫자... ]
    
    print(collections.Counter(ranks)) # Results vary due to random seeding and very small corpus
    # train_corpus[doc_id].words의 type은 str이다.
    
    print("=======================================생성된 모델 잘 나오는지 한번! 확인하는 부분 ==============")
    print('Document ({}): ≪{}≫\n '.format(doc_id, '?'.join(train_corpus[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        # list index out of range 에러 부분 발견 
        print(u'%s %s : ≪%s≫\n' % (label, sims[index],' '.join(test_corpus[sims[index][0]].words)))
       
    print("================================ pick a random doc from test corpus and see the results")   
    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus))
    # Compare and print the most/median/least similar documents from the train corpus
    #print('Train Document ID ({}): ≪{}≫\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print('Train Document ID ({}): ≪{}≫\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    sim_id = second_ranks[doc_id]
    print('Similar Document ID ({}): ≪{}≫\n'.format(sim_id, ' '.join(test_corpus[sim_id[0]].words)))
    print("\n ============================= DONE ================================================")


train()

# model load and test it 
model = Doc2Vec.load("not_preprocess.doc2vec")
#print("호잇",model.wv.syn0)
# 위와 코드가 겹치는 이유? 원래 모델 만들고 나서 테스트 해보는걸 동일하게 가져가기 때문이다.
ranks = []
first_ranks=[]
second_ranks = []
model.init_sims(replace=True)
# 단어 vecotor가져오는 과정 
vocab = model.wv.index2word
word_vectors= []
for v in vocab : 
    word_vectors.append(model.wv[v])

print("\n<<<<<<<<<<<<<<<<<<<<<<<<< word vector가져와보기!!!")
print(vocab[:10]) #token단위 단어 ex) ['성품을', '항상', '지니고', '대함', '마음으로', '친구들에게', '착하고', '잇으며,', '순수한', '진심어린']
print(vocab[0]) # 성품을 
#print(model.wv[vocab[0]]) # vector 값 보여지기 
#print(word_vectors[0]); print(model.wv['성품을']) # vecot 값 보여지기  
print(model.most_similar(u'성품을'))
print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<< DONE ")


print(" most_similar result")
print(model.docvecs.most_similar(3))


memory = []
for doc_id in range(len(train_corpus)):
    sims = model.docvecs.most_similar([model.docvecs[doc_id]], topn=len(model.docvecs))   
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    first_ranks.append(sims[0])
    second_ranks.append(sims[1])
    
print(collections.Counter(ranks)) # Results vary due to random seeding and very small corpus

print("======================================= TEST 입니다 =========================================")
print('Document ({}): 내용  : {}\n '.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s 내용 : %s \n' % (label, sims[index],' '.join(test_corpus[sims[index][0]].words)))

doc_id = random.randint(0, len(train_corpus))
# Compare and print the most/median/least similar documents from the train corpus
print('Train Document {}: 내용 : {}\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}, 내용 : {}\n'.format(sim_id, ' '.join(test_corpus[sim_id[0]].words)))
print("======================================= TEST 끄읏!!!! =========================================")

#######################################################################################################################
while True : 
    print('Document id ????? 0~{}(if u want to exit then press -1)\n'.format(len(train_corpus)-1))
    doc_id = int(input())    
    
    if doc_id == -1 : 
        print("===================================finish===================================:)")
        break;
    elif (doc_id >= len(train_corpus)) :
        print('The End')
        continue;
    else : 
        a = model.docvecs.doctag_syn0[doc_id]
        # Compare and print the most/median/least similar documents from the train corpus
        print('TEST Document ({}): 내용{}\n'.format(doc_id,' '.join(train_corpus[doc_id].words)))
        sim_id = second_ranks[doc_id]
        b= model.docvecs.doctag_syn0[sim_id[0]]
        #print('\n\n cosine similarity is {} '.format(cosine(a,b)))
        c = sim_id[0]
        print('Similar Document {}: 내용{}\n'.format(sim_id,' '.join(test_corpus[sim_id[0]].words)))

        
        print("===============================================  word mover's distance ===============================================")
        
        make_one = ' '.join(train_corpus[doc_id].words).split('. ') # test 이 문장이랑 유사한거 나와!
        make_two = ' '.join(train_corpus[sim_id[0]].words).split('. ') # train 이거다!!!
        #make_one = train_corpus[doc_id].words.split(' ')
        #make_two = train_corpus[sim_id[0]].words.split(' ')
        
        print(make_one)
        print(make_two)

        check_list = [(model.wmdistance(i.split(' '),j.split(' ')),i,j) for i in make_one for j in make_two]
        print(len(make_one), len(make_two), len(check_list))
        check_list = list(set(check_list))
        print(len(check_list))
        
        for ccc in check_list : #[(wmdistance, str1, str2),(..,..,..),(..,..,..)] 
            #print(ccc[0],ccc[1],ccc[2])
            if ccc[0]<0.2 :
                print("이 문장과 같은 또는 비슷한 문장은? " ,ccc[1],"\n wmdistance값은 {}이며 내용은,,".format(ccc[0]),ccc[2])
            elif ccc[0]>=0.2  and ccc[0]<0.5 :
                print("이 문장과 얼추  문맥 일치하는  문장은? " ,ccc[1],"\n wmdistance값은 {}이며 내용은,,".format(ccc[0]),ccc[2])
        
             
        '''
        make_one = ''.join(train_corpus[doc_id].words).split('. ') # test 이 문장이랑 유사한거 나와!
        make_two = ' '.join(train_corpus[sim_id[0]].words).split('. ') # train 이거다!!!
    
        check_list = [(model.wmdistance(i,j),i,j) for i in make_one for j in make_two]
        print(len(make_one), len(make_two), len(check_list))
        check_list = list(set(check_list))
        print(len(check_list))
        for ccc in check_list :
            if ccc[0]<0.2 :
                print("이 문장과 같은 또는 비슷한 문장은? " ,ccc[1],"\n wmdistance값은 {}이며 내용은,,".format(ccc[0]),ccc[2])
            elif ccc[0]>=0.2  and ccc[0]<0.5 :
                print("이 문장과 얼추  문맥 일치하는  문장은? " ,ccc[1],"\n wmdistance값은 {}이며 내용은,,".format(ccc[0]),ccc[2])
        '''
