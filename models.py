import numpy
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Embedding, Concatenate    

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers 
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.optimizers import *

import tensorflow as tf
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import InputLayer, Dense, Embedding, Reshape, BatchNormalization, Masking,Lambda  # my code!

import seq2tens  

npratio = 4
ls2t_size = 200
ls2t_order = 2
ls2t_depth = 3


class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        
        Q_seq = K.dot(Q_seq, self.WQ)  
        Q_seq = K.reshape(Q_seq, (-1, Q_seq.shape[1], self.nb_head, self.size_per_head))  # Q_seq:(None, 28, 20, 20)  
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        
        K_seq = K.dot(K_seq, self.WK)   # K_seq:(None, 28, 400)
        K_seq = K.reshape(K_seq, (-1, K_seq.shape[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))  
        
        V_seq = K.dot(V_seq, self.WV)   
        V_seq = K.reshape(V_seq, (-1, V_seq.shape[1], self.nb_head, self.size_per_head))  
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))   # V_seq:(None, 20, 28, 20)

        A = tf.matmul(Q_seq, K.permute_dimensions(K_seq, (0,1,3,2))) / self.size_per_head**0.5   # A (None, 20, 28, 28)
        
        A = K.permute_dimensions(A, (0,3,2,1))   
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))  
        A = K.softmax(A) # A:(None, 20, 28, 28)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])   
        O_seq = tf.matmul(A, V_seq)     # O_seq:(None, 20, 28, 20)
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, O_seq.shape[1], self.output_dim))    
        O_seq = self.Mask(O_seq, Q_len, 'mul')   # O_seq:(None, 28, 400)
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32') 
    user_vecs =Dropout(0.2)(vecs_input)   # (None, 30, 400)
    user_att = Dense(200,activation='tanh')(user_vecs)  # (None, 30, 200)
    user_att = tf.keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = tf.keras.layers.Dot((1,1))([user_vecs,user_att])  # user_vec (None, 400)
    model = Model(vecs_input,user_vec)
    return model


def get_doc_encoder():   
    sentence_input = Input(shape=(30,300), dtype='float32')
    droped_vecs = Dropout(0.2)(sentence_input)
    l_cnnt = tf.keras.layers.Dense(400, activation='relu')(droped_vecs)
    l_cnnt = Dropout(0.2)(l_cnnt)
    l_cnnt = Attention(20,20)([l_cnnt,l_cnnt,l_cnnt])  
    drop_cnnt = Dropout(0.2)(l_cnnt)
    title_vec = AttentivePooling(30,400)(drop_cnnt)    # title_vec:(None, 400)
    sentEncodert = Model(sentence_input, title_vec)
    return sentEncodert

    
def get_user_encoder():   
    news_vecs_input = Input(shape=(60,400), dtype='float32') 
    
    nlstlayer31 = seq2tens.layers.Time()(news_vecs_input)
    nlstlayer32 = seq2tens.layers.Difference()(nlstlayer31)
    nlstlayer33 = seq2tens.layers.LS2T(ls2t_size, ls2t_order, return_sequences=False, recursive_weights=True)(nlstlayer32)
    nlstlayer34 = Reshape((ls2t_order, ls2t_size,))(nlstlayer33)
    nlstlayer35 = BatchNormalization(axis=1)(nlstlayer34)
    user_vec1 = Reshape((ls2t_order * ls2t_size,))(nlstlayer35)  # user_vec1: (None, 400)
    user_vec1 = tf.keras.layers.Reshape((1,400))(user_vec1)  # user_vec1: (None, 1, 400)
    
    user_vecs2 = Lambda(lambda x:x[:,-20:,:])(news_vecs_input)
    user_vec2 = GRU(400)(user_vecs2)  
    user_vec2 = tf.keras.layers.Reshape((1,400))(user_vec2) # user_vec2: (None, 1, 400)
    
    user_vecs = tf.keras.layers.Concatenate(axis=-2)([user_vec1,user_vec2])  
    vec = AttentivePooling(2,400)(user_vecs)    # vec:(None, 400)
        
    sentEncodert = Model(inputs=news_vecs_input, outputs = vec)
    return sentEncodert


def get_model(lr,delta,title_word_embedding_matrix):
    doc_encoder = get_doc_encoder()  
    user_encoder = get_user_encoder()
    
    title_word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], 300, weights=[title_word_embedding_matrix],trainable=False)
    
    click_title = Input(shape=(60,30),dtype='int32')
    can_title = Input(shape=(1+npratio,30),dtype='int32')  
    
    click_word_vecs = title_word_embedding_layer(click_title)   # (None, 50, 30, 300)
    can_word_vecs = title_word_embedding_layer(can_title)
    
    click_vecs = TimeDistributed(doc_encoder)(click_word_vecs)  # (None, 50, 400)
    can_vecs = TimeDistributed(doc_encoder)(can_word_vecs)
    
    user_vec = user_encoder(click_vecs)
    
    scores = tf.keras.layers.Dot(axes=-1)([user_vec,can_vecs]) 
    logits = tf.keras.layers.Activation(tf.keras.activations.softmax,name = 'recommend')(scores)     
    
    model = Model([can_title,click_title],logits) 
    
    model.compile(loss=['categorical_crossentropy'],    
                  optimizer=SGD(lr=lr,clipvalue = delta),  
                  metrics=['acc'])   
    
    news_input = Input(shape=(30,),dtype='int32')
    news_word_vecs = title_word_embedding_layer(news_input)
    news_vec = doc_encoder(news_word_vecs)
    news_encoder = Model(news_input,news_vec)    
    
    return model, doc_encoder, user_encoder, news_encoder