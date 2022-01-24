import nltk 
from nltk.tokenize import word_tokenize
import csv
import datetime
import time
import json
import itertools
import random
import os
import numpy as np
from tensorflow.keras.layers import Embedding


MAX_SENTENCE = 30
MAX_ALL = 60
npratio = 4

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)
        
def read_news(root_data_path,modes):
    news={}
    category=[]
    subcategory=[]
    news_index={}
    index=1
    word_dict={}
    word_index=1
    
    for mode in modes:   
        with open(os.path.join(root_data_path,mode,'movies.tsv')) as f:
            lines = f.readlines()
        for line in lines:
            splited = line.strip('\n').split('\t')
            doc_id = splited[0]
            vert = splited[1].split(' ')[0]
            subvert = splited[1].split(' ')[1]
            title = splited[2]
            # print(title)
            if doc_id in news_index:
                continue
            news_index[doc_id]=index  # reate a dictionary for items,key:doc_id,value:index
            index+=1
            category.append(vert)
            subcategory.append(subvert)
            # title = title.lower()
            # title = word_tokenize(title)
            news[doc_id]=[vert,subvert,title]
            title = title.split(' ')
            for word in title:
                # word = word.lower()
                if not(word in word_dict):
                    word_dict[word]=word_index
                    word_index+=1
    category=list(set(category))
    subcategory=list(set(subcategory))
    category_dict={}
    index_category = 1
    for c in category:
        category_dict[c] = index_category
        index_category += 1
    subcategory_dict={}
    index_subcategory = 1
    for c in subcategory:
        subcategory_dict[c] = index_subcategory
        index_subcategory += 1
    return news,news_index,category_dict,subcategory_dict,word_dict


def get_doc_input(news,news_index,category,subcategory,word_dict):
    news_num = len(news)+1  
    news_title = np.zeros((news_num, MAX_SENTENCE),dtype='int32')
    news_vert = np.zeros((news_num,),dtype='int32')
    news_subvert = np.zeros((news_num,),dtype='int32')
    for key in news:    
        vert,subvert,title = news[key]
        title = title.split(' ')
        doc_index = news_index[key]
        news_vert[doc_index] = category[vert]
        news_subvert[doc_index]=subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE,len(title))):
            news_title[doc_index,word_id] = word_dict[title[word_id]]
            # news_title[doc_index,word_id] = word_dict[title[word_id].lower()]
        
    return news_title,news_vert,news_subvert

    
def load_matrix(embedding_path,word_dict):  
    embedding_matrixEm = Embedding(len(word_dict)+1, 300, trainable=False)
    embedding_matrix0 = np.zeros((len(word_dict)+1,300))
    for i in range(len(word_dict)+1):
        embedding_matrix0[i] = embedding_matrixEm(i)
    embedding_matrix1 = np.zeros((len(word_dict)+1,300))
    have_word=[]
    all_word = [k for k in word_dict]
    word_glove_dict = {}
    word_glove = set()
    with open(os.path.join(embedding_path, 'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            # print(l[1:])
            k = l[0]
            # print(k)
            v = l[1:]
            word_glove_dict[k] = v
            word_glove.add(k)
            # print(word_dict)
            line = f.readline()
    for k,v in word_dict.items():
        if k in word_glove:
            index = word_dict[k]
            v = word_glove_dict[word]
            # tp = [float(x) for x in l[1:]]
            embedding_matrix1[index]=np.array(v)
            have_word.append(word)
        else:
            index = word_dict[k]
            embedding_matrix1[index]=embedding_matrix0[index]
    embedding_matrix = (embedding_matrix0 + embedding_matrix1)/2      
    return embedding_matrix,have_word


def read_clickhistory(root_data_path,mode,filename):
    
    lines = []
    userids = {}
    uid_table = {}
    with open(os.path.join(root_data_path,mode, filename)) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,_,click,imp = lines[i].strip().split('\t')
        true_click = click.split(',')
        assert not '' in true_click
        if not uid in userids:
            uid_table[len(userids)] = uid
            userids[uid] = []  # A user has more than one record
        userids[uid].append(i)
        imp = imp.split(',')
        # print(imp)
        pos = []
        neg = []
        for beh in imp:
            nid, label = beh.split('+++')
            if label == '0':
                neg.append(nid)
            else:
                pos.append(nid)
        sessions.append([true_click,pos,neg])
    return sessions,userids,uid_table


def parse_user(session,news_index):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_ALL),dtype='int32'),}  
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg = session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) > MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click=[0]*(MAX_ALL-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user


def get_train_input(session,uid_click_talbe,news_index):
    inv_table = {}
    user_id_session = {}

    for uid in uid_click_talbe:
        user_id_session[uid] = []
        for v in uid_click_talbe[uid]:
            inv_table[v] = uid
    
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)                
            user_id_session[inv_table[sess_id]].append(len(sess_pos)-1)
            
    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    # print(user_id_session)
    return sess_all, user_id, label, user_id_session


def get_test_input(session,news_index):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,