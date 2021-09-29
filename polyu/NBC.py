import array
import math

from polyu.NBC1 import GaussianNBMMM

"""
NBC为一个类，输入为4个参数

"""
from sklearn.pipeline import Pipeline
import numpy

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
import numpy as np
from numpy import ndarray, exp, pi, sqrt, log, sum

class NBC:
    def __init__(self, feature_types, num_classes, landa = 1 * math.e ** -10):
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.landa = landa
        self.avg = None
        self.var = None
        self.prior = None

    def fit(self, Xtrain, ytrain):
        '''
        Xtrain is the four features , y is the lable of every row,
        first we need to ues ytrain to get the priori probability of THREE LABELS
        '''
        self.prior = self.get_y_pri(ytrain)

        Xy = np.c_[np.array(Xtrain), np.array(ytrain).T]
        # the four features average value of the three labels
        self.avg = self.get_x_avg(Xtrain, ytrain)
        # the four features var value of the three labels
        # var = power(std, 2)
        self.var = self.get_x_var(Xtrain, ytrain)
        # use avg and var to get the likelihood of each label(Gaussian distribution)
        # y_likelihood = self.get_y_likelyhood(x_avg, x_var)

        # self.likelihood = self.get_likelihood(Xtrain, ytrain)






        clf = GaussianNB()
        pass

    def predict_prob(self, Xtest):
        likelihood = np.apply_along_axis(self.get_likelihood, axis=1, arr=Xtest)
        probs = self.prior * likelihood
        # 按行相加
        # probs_sum = probs.sum(axis=1)
        # probs_sum[:, None] 相当于转置
        # res = probs / probs_sum[:, None]
        return probs

    def predict(self, Xtest):
        res = self.predict_prob(Xtest)
        res1 = res.argmax(axis=1)
        return res1

    def get_count(self, ytrain, c):
        count = 0
        for y in ytrain:
            if y == c:
                count += 1
        return count

    def get_y_pri(self, ytrain):
        ytrain_len = len(ytrain)
        pri_p_0 = self.get_count(ytrain, 0) / ytrain_len
        pri_p_1 = self.get_count(ytrain, 1) / ytrain_len
        pri_p_2 = self.get_count(ytrain, 2) / ytrain_len
        return np.array([pri_p_0, pri_p_1, pri_p_2])

    def get_x_var(self, Xtrain, ytrain):
        res = []
        for i in range(self.num_classes):
            res.append(Xtrain[ytrain == i].var(axis=0))
        return np.array(res)

    def get_likelihood(self, para: array) -> array:

        ztfb = (1 / sqrt(2 * pi * self.var) * exp(-1 * (para - self.avg)**2 / (2 * self.var)))
        res = (log(ztfb)).sum(axis=1)

        return res
        # res = []
        # for i in range(self.num_classes):
        #     curr_arr = Xtrain[ytrain == i]
        #     log_val = []
        #     for j in range(len(self.feature_types)):
        #         feature_col = curr_arr[:, j]
        #         col_log_likelihood_sum = 0
        #         for val in feature_col:
        #             col_log_likelihood_sum += self.get_single_likelihood(val, i, j)
        #         log_val.append(col_log_likelihood_sum)
        #     res.append(log_val)
        # return np.array(res)

    def get_single_likelihood(self, val, i, j):
        return math.log( (1 / math.sqrt(2 * math.pi * self.var[i][j])) * math.exp(-1 * math.pow(val - self.avg[i][j], 2) / 2 * self.var[i][j]))


    def get_x_avg(self, Xtrain, ytrain):
        res = []
        for i in range(self.num_classes):
            res.append(Xtrain[ytrain == i].mean(axis=0))
        return np.array(res)


from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris['data'], iris['target']

# shape纬度
N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
nbc.fit(Xtrain, ytrain)
res = nbc.predict(Xtest)
ytest = ytest
test_accuracy = np.mean(res == ytest)
# ab = np.c_[np.array(res), np.array(ytest), np.array(res - ytest)]
# cp = np.c_(res, ytest)
nbc1 = GaussianNBMMM()
nbc1.fit(Xtrain, ytrain)
res1 = nbc1.predict(Xtest)
test_accuracy1 = np.mean(res1 == ytest)
s = 1