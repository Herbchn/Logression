import theano
import theano.tensor as T
from theano import shared
import numpy as np

class Regression:
    """
    Regression implements based on theano.
    """
    def __init__(self, dim, cost="CrossEnp", w=None, b=None, step = 0.01, regular = 0.01):
        rng = np.random

        # define the input
        x = T.dmatrix(name="x")
        y = T.dvector(name="y")

        # initial the parameter
        self.dim = dim
        self.step = 0.1

        if w == None:
            self.w = shared(rng.randn(dim), name="w")
        else:
            self.w = shared(w, name = "w")

        if b == None:
            self.b = shared(0., name = "b")
        else:
            self.b = shared(b, name = "b")

        # define the regression(g), prediction, error, cost
        self.g = 1./(1. + T.exp(-T.dot(x,self.w) - self.b))
        self.prediction = self.g > 0.5
        self.error = -y*T.log(self.g) - (1-y)*T.log(1-self.g)
        cost = self.error.mean() + regular * (self.w ** 2).sum()

        # compute gradient
        gw, gb = T.grad(cost, [self.w, self.b])

        # define reg, train, test
        self.predict = theano.function([x], self.prediction)
        self.train = theano.function([x, y], self.error, updates = [(self.w, self.w - step * gw), (self.b, self.b - step * gb)])
        self.test = theano.function([x,y], [self.prediction, 1 - T.mean(T.neq(self.prediction, y))])

    def fun_predicton(self, x):
        return self.predict(x)

    def fun_train(self, x, y, epoches):
            print "training ..."
            for i in range(epoches):
                    self.train(x,y)
            print "training end"

    def fun_test(self, x, y):
            print "testing ..."
            return self.test(x,y)
