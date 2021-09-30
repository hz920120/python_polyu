import numpy as np
from numpy import exp, pi, sqrt, log, e

from sklearn.datasets import load_iris


class NBC:
    def __init__(self, feature_types, num_classes, landa=1 * e ** -6):
        """

        Args:
            feature_types:
            num_classes:
            landa: avoid the scenario of log0, defalt 1e-6
        """
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.landa = landa
        self.avg = None
        self.var = None
        self.prior = None

    def fit(self, Xtrain, ytrain):
        """
        Xtrain is the four features , y is the lable of every row,
        we need to use parameters to get some CONSTANTS(average, variance, prior probability)
        in order to predict test datasets
        Args:
            Xtrain:
            ytrain:

        Returns:

        """
        self.prior = self.get_y_pri(ytrain)
        # the four features average values of the three labels
        self.avg = self.get_x_avg(Xtrain, ytrain)
        # the four features's var values of the three labels
        # var = power(std, 2)
        self.var = self.get_x_var(Xtrain, ytrain)

    def predict_prob(self, Xtest):
        """
        calculate the probability of every row in the test dataset
        in order to choose the closest label of this row
        Args:
            Xtest:

        Returns:
            array
        """
        # apply_along_axis means cut the Xtest into rows in order to calculate easier the likelihood
        likelihood = np.apply_along_axis(self.get_likelihood, axis=1, arr=Xtest)
        return self.prior * likelihood

    def predict(self, Xtest):
        """
        choose the largest probability as the label of row, return the label array
        Args:
            Xtest:

        Returns:
            array
        """
        return self.predict_prob(Xtest).argmax(axis=1)

    def get_count(self, ytrain, c):
        """
        get total number of every label in thetrain dataset
        Args:
            ytrain:
            c: class lable

        Returns:
            int count
        """
        count = 0
        for y in ytrain:
            if y == c:
                count += 1
        return count

    def get_y_pri(self, ytrain):
        """
        get prior probability of all labels
        Args:
            ytrain:

        Returns:
            array
        """
        ytrain_len = len(ytrain)
        res = []
        for y in range(self.num_classes):
            pri_p = self.get_count(ytrain, y) / ytrain_len
            res.append(pri_p)
        return np.array(res)

    def get_x_var(self, Xtrain, ytrain):
        """
        get variance of every feature in the train dataset,
        the result is necessary for predicting test dataset
        Args:
            Xtrain:
            ytrain:

        Returns:
            array
        """
        res = []
        for i in range(self.num_classes):
            res.append(Xtrain[ytrain == i].var(axis=0))
        return np.array(res)

    def get_likelihood(self, label_row):
        """
        get likelihood probability of every row of test dataset

        we add landa parameter manually to avoid the computation result of Gaussian distribution may be zero
        Args:
            label_row:

        Returns:
            array
        """

        # landa parameter is very important
        gauss_dis = (1 / sqrt(2 * pi * self.var) * exp(-1 * (label_row - self.avg) ** 2 / (2 * self.var))) + self.landa
        # log(abc) = loga + logb + loc
        return (log(gauss_dis)).sum(axis=1)

    def get_x_avg(self, Xtrain, ytrain):
        """
        get average of every feature in the train dataset,
        the result is necessary for predicting test dataset
        Args:
            Xtrain:
            ytrain:

        Returns:
            array
        """
        res = []
        for i in range(self.num_classes):
            res.append(Xtrain[ytrain == i].mean(axis=0))
        return np.array(res)


iris = load_iris()
X, y = iris['data'], iris['target']

N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
nbc.fit(Xtrain, ytrain)
yhat = nbc.predict(Xtest)
test_accuracy = np.mean(yhat == ytest)

print("Congrats! Accuracy is %.3f%%! Excellent model!" % (test_accuracy * 100))
