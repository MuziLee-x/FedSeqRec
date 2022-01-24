import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import *
from preprecoess import *
from generator import *
from models import *

client_data_path = "/home/lab31/FreedomLi/FedSeqRec(movie)/clients/MoviesData/"

def add_noise(weights,lambd):
    for i in range(len(weights)):
        weights[i] += np.random.laplace(scale = lambd,size=weights[i].shape)
    return weights
    
    
def client_updata(model,sample,click,label, batchsize):
    g = model.fit([sample,click],label,batch_size = batchsize,verbose=False)
    loss = g.history['loss'][0]
    acc = g.history['acc'][0]
    return loss,acc
    
    

def fed_single_update(model, doc_encoder, user_encoder, clients, lambd, batchsize, news_encoder, news_title, test_user, test_impressions, test_userids, count, alpha):
    
    clients = ['MovieLens', 'Tweetings']
    weight = [0.4739437986842115, 0.5360562013157885]   # dataprepare
    
    all_news_weights = []
    all_user_weights = []
    old_news_weight = doc_encoder.get_weights()
    old_user_weight = user_encoder.get_weights()
    
    sample_nums = []
    
    loss = []
    res_nums = []
    gra_nums = []
    att = []
    similarity = []
    simi = []
    for clientid in clients:
        doc_encoder.set_weights(old_news_weight)
        user_encoder.set_weights(old_user_weight)
        
        sample = np.load(client_data_path + f'{clientid}_Sample.npy')   
        click = np.load(client_data_path + f'{clientid}_Click.npy')   
        label = np.load(client_data_path +  f'{clientid}_Label.npy')  
        
        l,_ = client_updata(model,sample,click,label, batchsize)
        loss.append(l)
        news_weight = doc_encoder.get_weights()      
        
        user_weight = user_encoder.get_weights()
        if lambd>0:
            news_weight = add_noise(news_weight,lambd)
            user_weight = add_noise(user_weight,lambd)   
        all_news_weights.append(news_weight)
        all_user_weights.append(user_weight)
        sample_nums.append(label.shape[0])

    sample_nums = np.array(sample_nums)
    att2 = sample_nums/sample_nums.sum()
    
    att1 = np.array(weight)
    att = alpha*att1 + (1-alpha)*att2

    doc_weights = [np.average(weights, axis=0,weights=weight) for weights in zip(*all_news_weights)]
    user_weights = [np.average(weights, axis=0,weights=weight) for weights in zip(*all_user_weights)]

    doc_encoder.set_weights(doc_weights)
    user_encoder.set_weights(user_weights)
    
    loss = np.array(loss).mean()

    return loss