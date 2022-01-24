#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os

import tensorflow as tf
import numpy as np

from utils import *
from preprecoess import *
from generator import *
from models import *
from fl_training import *


# # Cluster split
root_data_path = ...
client_data_path =  ...
embedding_path = ...

news,news_index,category_dict,subcategory_dict,word_dict = read_news(root_data_path,['train','val'])
news_title,news_vert,news_subvert = get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict)
title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)

test_session, test_uid_click,test_uid_table = read_clickhistory(root_data_path,'val','All_MLTW_val_behaviors.tsv')

test_user = parse_user(test_session,news_index)
test_impressions, test_userids = get_test_input(test_session,news_index)

lr = 0.40   # 
# lr = 0.30  
delta = 0.05
lambd = 0.020
ratio = 0.02
# clients = 16  # MoocData
clients = ['MovieLens', 'Tweetings']  # Movies
batchsize = 32  # my code!!
alpha = 0.3
# alpha = 1


model, doc_encoder, user_encoder, news_encoder = get_model(lr,delta,title_word_embedding_matrix)
print(model.summary())

Res = []
Loss = []
count = 0
while count<=500:
    loss = fed_single_update(model, doc_encoder, user_encoder, clients, lambd, batchsize, news_encoder, news_title, test_user, test_impressions, test_userids, count, alpha)   # the Author's code!!
    Loss.append(loss)
    print(loss)
    if count % 1 == 0:
        news_scoring = news_encoder.predict(news_title,verbose=0)
        user_generator = get_hir_user_generator(news_scoring,test_user['click'],64)
        user_scoring = user_encoder.predict_generator(user_generator,verbose=0)  # Model.predict also supports generators.
        g = evaluate(user_scoring,news_scoring,test_impressions)
        Res.append(g)
        print(g)
        with open('FedSeqRec_privacy(0.020)_Movie.json','a') as f: 
            s = json.dumps(g) + '\n'
            f.write(s)
    count += 1
    