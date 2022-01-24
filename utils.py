from sklearn.metrics import roc_auc_score
import numpy as np

def dcg_score(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def evaluate(user_scorings,news_scorings,Impressions):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    nDCG20 = []
    nDCG50 =[]
    for i in range(len(Impressions)):
        docids = Impressions[i]['docs']
        labels = Impressions[i]['labels']
        uv = user_scorings[i]
        
        docids = np.array(docids,dtype='int32')
        nv = news_scorings[docids]
        score = np.dot(nv,uv)
        auc = roc_auc_score(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
        ndcg20 = ndcg_score(labels,score,k=20)
        ndcg50 = ndcg_score(labels,score,k=50)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
        nDCG20.append(ndcg20)
        nDCG50.append(ndcg50)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)
    nDCG20 = np.array(nDCG20)
    nDCG50 = np.array(nDCG50)
    
    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    nDCG20 = nDCG20.mean()
    nDCG50 = nDCG50.mean()
    
    return AUC, MRR, nDCG5, nDCG10, nDCG20, nDCG50