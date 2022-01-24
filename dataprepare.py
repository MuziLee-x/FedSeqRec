import json
import pandas as pd
import random
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time, datetime
import math

'''
Combine the test set of the two data sets
'''

MovieLens_path = ...
Tweetings_path = ...
       

file = ['Tweetings_train_behaviors.tsv','MovieLens_train_behaviors.tsv']

session_Tw = []
click_Tw =[]
user_Tw = []
user_click_Tw = []
with open('Tweetings_train_behaviors.tsv', 'r') as f:
    line = f.readline()
    while line:
        tmp = {}
        arr = line.strip('\n').split('\t')
        # print(arr)
        click = arr[3].split(',')
        # print(click)
        session_Tw.append(arr[0])
        user_Tw.append(arr[1])
        click_Tw.append(len(click)+1)
        tmp[arr[1]] = len(click)
        user_click_Tw.append(tmp)
        line = f.readline()
print(len(session_Tw))  
print(len(click_Tw))  
print(len(user_Tw))  
print(len(user_click_Tw))  
user_click_Twtmp = {}
for _ in user_click_Tw:
    for k, v in _.items():
        user_click_Twtmp.setdefault(k, []).append(v)
print('--------------------')
print('Tw:the average session length is: ',np.mean(np.array(click_Tw)))     
user_set_Tw = set(user_Tw)
user_set_Tw = list(user_set_Tw)
user_avgclick_Tw = []
for i in range(len(user_set_Tw)):
    if user_set_Tw[i] in user_click_Twtmp.keys():
        click = user_click_Twtmp[user_set_Tw[i]]
        click_avg = np.mean(np.array(click))
        user_avgclick_Tw.append(click_avg)
print('Tw:the average user click length is: ',np.mean(np.array(user_avgclick_Tw)))  # 5.294986354225485
print('*********************************')


session_Ml = []
click_Ml =[]
user_Ml = []
user_click_Ml = []
with open('MovieLens_train_behaviors.tsv', 'r') as f:
    line = f.readline()
    while line:
        tmp = {}
        arr = line.strip('\n').split('\t')
        # print(arr)
        click = arr[3].split(',')
        # print(click)
        session_Ml.append(arr[0])
        user_Ml.append(arr[1])
        click_Ml.append(len(click)+1)
        tmp[arr[1]] = len(click)
        user_click_Ml.append(tmp)
        line = f.readline()
print(len(session_Ml))  
print(len(click_Ml))  
print(len(user_Ml))  
print(len(user_click_Ml))  
user_click_Mltmp = {}
for _ in user_click_Ml:
    for k, v in _.items():
        user_click_Mltmp.setdefault(k, []).append(v)
print('--------------------')
print('Ml:the average session length is: ',np.mean(np.array(click_Ml)))   
user_set_Ml = set(user_Ml)
user_set_Ml = list(user_set_Ml)
user_avgclick_Ml = []
for i in range(len(user_set_Ml)):
    if user_set_Ml[i] in user_click_Mltmp.keys():
        click = user_click_Mltmp[user_set_Ml[i]]
        click_avg = np.mean(np.array(click))
        user_avgclick_Ml.append(click_avg)
print('Ml:the average user click length is: ',np.mean(np.array(user_avgclick_Ml))) 
print('*********************************')


session_server = []
click_server =[]
user_server = []
user_click_server = []
with open('All_MLTW_val_behaviors.tsv', 'r') as f:
    line = f.readline()
    while line:
        tmp = {}
        arr = line.strip('\n').split('\t')
        # print(arr)
        click = arr[3].split(',')
        # print(click)
        session_server.append(arr[0])
        user_server.append(arr[1])
        click_server.append(len(click)+1)
        tmp[arr[1]] = len(click)
        user_click_server.append(tmp)
        line = f.readline()
print(len(session_server)) 
print(len(click_server))  
print(len(user_server)) 
print(len(user_click_server))  
user_click_servertmp = {}
for _ in user_click_server:
    for k, v in _.items():
        user_click_servertmp.setdefault(k, []).append(v)
print('--------------------')
print('server:the average session length is: ',np.mean(np.array(click_server)))   
user_set_server = set(user_server)
user_set_server = list(user_set_server)
user_avgclick_server = []
for i in range(len(user_set_server)):
    if user_set_server[i] in user_click_servertmp.keys():
        click = user_click_servertmp[user_set_server[i]]
        click_avg = np.mean(np.array(click))
        user_avgclick_server.append(click_avg)
print('server:the average user click length is: ',np.mean(np.array(user_avgclick_server))) 
print('*********************************')

TwFeature = [np.mean(np.array(click_Tw)), np.mean(np.array(user_avgclick_Tw))]
MlFeature = [np.mean(np.array(click_Ml)), np.mean(np.array(user_avgclick_Ml))]
serverFeature = [np.mean(np.array(click_server)), np.mean(np.array(user_avgclick_server))]
simTwServer = np.linalg.norm(np.array(TwFeature)-np.array(serverFeature))
simMlServer = np.linalg.norm(np.array(MlFeature)-np.array(serverFeature))
print('The similarity of Tw and Server is: ',(simTwServer,math.log(simTwServer)))
print('The similarity of Ml and Server is: ',(simMlServer,math.log(simMlServer)))
weight = math.log(simMlServer)/(math.log(simTwServer)+math.log(simMlServer))
att = [weight, 1-weight]
print(att)
