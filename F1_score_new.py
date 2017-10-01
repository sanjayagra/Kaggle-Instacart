
# coding: utf-8

# In[14]:

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime


# In[15]:

path = '/Users/sanjayagrawal/Downloads/'
data1 = pd.read_csv(path + 'instacart_data.csv')
products = list(data1['product_id'].unique())
data1.shape


# In[16]:

data_prod = data1.groupby(['order_id'])['product_id'].apply(list).reset_index()
data_pred = data1.groupby(['order_id'])['prediction'].apply(list).reset_index()
data = data_prod.merge(data_pred, on='order_id', how = 'inner')
data.head(2)


# In[17]:

data['tuple1'] = data.apply(lambda x : list(zip(x['product_id'], x['prediction'])), axis=1)
data['dict1'] = data['tuple1'].map(lambda x : {i[0]:i[1] for i in x})
data.head(2)


# In[ ]:

class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)


def print_best_prediction(Probs, pNone=None):
#     print("Maximize F1-Expectation")
#     print("=" * 23)
#     print (Probs)
#     P = np.sort(P)[::-1]
#     n = P.shape[0]
#     L = ['L{}'.format(i + 1) for i in range(n)]
    k = sorted([(o,s) for o,s in Probs.items()], key = lambda x : x[1], reverse = True)
    L = [off for off, score in k]
    P = [score for off, score in k]
    n = len(L)
    P = np.array(P)

    if pNone is None:
#         print("Estimate p(None|x) as (1-p_1)*(1-p_2)*...*(1-p_n)")
        pNone = (1.0 - P).prod()

    PL = ['p({}|x)={}'.format(l, p) for l, p in zip(L, P)]
#     print("Posteriors: {} (n={})".format(PL, n))
#     print("p(None|x)={}".format(pNone))

    opt = F1Optimizer.maximize_expectation(P, pNone)
    best_prediction = ['None'] if opt[1] else []
    best_prediction += (L[:opt[0]])
    f1_max = opt[2]

#     print("Prediction {} yields best E[F1] of {}\n".format(best_prediction, f1_max))
    return best_prediction

if __name__ == '__main__':
    
    data['products'] = data.dict1.map(print_best_prediction)
    
#     probs = {"A":0.15, "B":0.1, "C":0.2, "D":0.5}
#     print_best_prediction(probs)


# In[13]:

data = data[['order_id', 'products']]
data.head(50)
data.to_csv(path + 'final.csv', index=False)

