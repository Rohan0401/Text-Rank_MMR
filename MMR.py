import numpy as np
import glob
import re
#import os
#import pandas as pd
import math
from collections import Counter

#from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from multinomial_naive_bayes import MultinomialNaiveBayes

#Global variables
j = 0
k = 0
count = 0
# Sentence List
s_list = {}
document= {}
lamb = 0
#This function returns bag of words for each class.
def BagofWords(x):
            s_final=[] #Final Sentences
            s_stop_words=[] #Sentences after removing Stop words
            #unique_list=[]
            sentence_list = []
            senlist = []
            files=0
            ct = 0
            #count = 0
            for line in x:
                 handle = open(line)
                 vocab = handle.read()
                 vocab = re.sub(r'\w*\d\w*','', vocab).strip()
                 vocab = re.sub('<DOC>','',vocab).strip()
                 vocab = re.sub('<DOCNO>','',vocab).strip()
                 vocab = re.sub('<DOCTYPE>','',vocab).strip()
                 vocab = re.sub('<DOC>','',vocab).strip()
                 vocab = re.sub('\n','',vocab).strip()
                 vocab = re.sub('</DOCNO>','',vocab).strip()
                 vocab = re.sub('</DOCTYPE>','',vocab).strip()
                 vocab = re.sub('</DOC>','',vocab).strip()
                 vocab = re.sub('<TXTTYPE>','',vocab).strip()
                 vocab = re.sub('</TXTTYPE>','',vocab).strip()
                 vocab = re.sub('<TEXT>','',vocab).strip()
                 vocab = re.sub('</TEXT>','',vocab).strip()
                
                 #vocab = re.sub('\'','',vocab).strip()
                 #vocab = re.sub('','',vocab).strip()
                 sent_tokenize_list = sent_tokenize(vocab)
                 global count                     
             #    for line1 in sent_tokenize_list:
                 s_list[count] = sent_tokenize_list
                 

               #  vocab = re.sub('[^A-Za-z]+', ' ', str(s_list[count]))
                 
                 word_tokenize_list = word_tokenize(str(s_list[count]))
                 
                 lower_list = [y.lower() for y in word_tokenize_list]
                 #print len(lower_list)
                 #print lower_list
                 stopset = stopwords.words('english')
                 filtered_word_list = [word for word in lower_list if word not in stopset]
                 
                 s_stop_words = [word for word in filtered_word_list if len(word) >= 3]
                 
                 
                 #unique_list= np.unique(s_stop_words)
                 #print unique_list
                 s_final.extend(s_stop_words)
                 sentence_list.append(s_stop_words)
                 #document[files] = s_final
                 senlist.extend(s_list[files])
                 #print "End of line \n"
                 count = count + 1
                 if count > 0:

                     if count % 10 == 0:
                         document[ct] = (senlist)
                         senlist = []
                         ct += 1
                 files = files+1
                 
            return s_final,files
            

fil1 = glob.glob('/home/rohan/Downloads/Final_Project/Documents/d*t/*')
dat_1 , files_1 = BagofWords(fil1)
#uniq_1= len(dat_1)
#print dat_1
#print len(dat_1)
#print files_1

#def MMR():


meta_doc = {}
temp = []
ct = 10
k = 0

#for i in range(len(document)):
#        temp.extend(document[i])
#        if i > 0:
#            if i % ct == 0:
#                meta_doc[k] = temp
#                temp = []
#                k += 1

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf_vectorizer = TfidfVectorizer()
#
##for i in range(len(meta_doc)):
#    
#tfidf_matrix = tfidf_vectorizer.fit_transform(document[1])
#
MMR=[]
S = ''
target = open('mysum.txt','w')
#tfidf_ofS = tfidf_vectorizer.fit_transform(S)

l1 = document[1][1]
l1 = l1.split()
strl2 = ''.join(str(w) for w in document[1])
l2 = strl2.split()



def build_vector(iterable1, iterable2):
    counter1 = Counter(iterable1)
    counter2 = Counter(iterable2)
    all_items = set(counter1.keys()).union(set(counter2.keys()))
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2
v1, v2 = build_vector(l1, l2)
#xx = []
#for i in range(len(document)):
def cosim(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)
#xx = []
senSet = []
for k in range(10):
    lamb = (float(k))/10
    print lamb
    for i in range(len(document)):
        S = S + ''.join(document[i][i])
        strl2 = ''.join(str(w) for w in document[i])
        l2 = strl2.split()
        for j in range(len(document[i])):
          if j > 1:    
              l1 = document[i][j]
              l1 = l1.split()
              if not senSet:
                  senSet.append(document[i][1])
                  
              v1,v2 = build_vector(l1,l2)
              firstTerm = lamb * cosim(v1,v2)
              maxSim2 = []
              for l in range(len(senSet)):
                sen = str(senSet[l].split())
                l2new = ''.join(str(w) for w in S)
                l2new = l2new.split()
                v1,v2 = build_vector(l1,l2new)
                maxSim2.append(cosim(v1,v2))
                
    
                    
    
    
                maxSimVal2 = max(maxSim2) 
                secondTerm = (1-lamb)*(maxSimVal2)
                if firstTerm - secondTerm > 0.1:
                    if len(S.split() + str(document[i][j]).split())<100: 
                        S = S + ''.join(document[i][j])    
              
              
              #print firstTerm - secondTerm
              
       
       
    
        print i,S
        target.write(str(i))
        print "\n"
        target.write(S)
        target.write('\n')
        print len(S.split())
        target.write(str(len(S.split())))
        print "\n"
        target.write('\n')
        S = ''    
                
    #secondTerm = 0.5 * cosim(document[1][1])        
    #MMR = firstTerm - secondTerm
   # if MMR > 0.4:
    #    S.append(document[1][j]) 
    
mmrscore=[]
#for j in range(len(document[1])):
#    MMR[j-1][j-1] = 0 
#    mmrscore.append(max(MMR[j-1]))
#        #print MMR
sentences = []
doc = {}
#
print S

#print type(s_list.get(1))
for senListIndex in range(len(s_list)):
    for i in range(len(s_list.get(senListIndex))):
    #sentences[i]= 
        sentences.append(s_list.get(senListIndex)[i])
   # print sentences        
doc[senListIndex] = sentences       
#    document.extend(str(sentences[i]))

#print "stopwords are:\n"
#stopset = stopwords.words('english')
#print stopset