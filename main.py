import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt


def generate_2DMoon(num=200, noise = 0.1):
    return ds.make_moons(num, noise = noise)

def generate_random(num = 20000, dim = 78):
    return [np.random.randn(num, dim), np.random.randint(size = num, low=0, high=2)]

def draw_scattrer(datas, plt, stya = "*", styb = "o"):
    x0 = [datas[0][i] for i in range(len(datas[1])) if datas[1][i] == 0]
    x1 = [datas[0][i] for i in range(len(datas[1])) if datas[1][i] == 1]
    plt.scatter([xi[0] for xi in x0], [xi[1] for xi in x0], marker = "*")
    plt.scatter([xi[0] for xi in x1], [xi[1] for xi in x1], marker = "o")

if __name__ == "__main__":
    execfile("tra_ML/Regression.py")

    # datas = generate_random(num = 300, dim = 2)
    datas = generate_2DMoon(10000)
    train_datas = [datas[0][:9950], datas[1][:9950]]
    test_datas = [datas[0][9950:], datas[1][9950:]]


    reg = Regression(2)
    reg.fun_train(datas[0], datas[1], 10000)
    res_test = reg.fun_test(test_datas[0], test_datas[1])
    print res_test[1]

    # draw

    x0 = np.array([x[0] for x in datas[0]])
    x1 = np.array([x[1] for x in datas[0]])

    x0_min, x0_max = x0.min(), x0.max()
    x1_min, x1_max = x1.min(), x1.max()
    xx0, xx1 = np.meshgrid(np.arange(x0_min - 0.5, x0_max + 0.5, 0.1), np.arange(x1_min - 0.5, x1_max + 0.5, 0.1))

    yy = np.reshape(reg.fun_predicton(np.c_[xx0.ravel(), xx1.ravel()]), xx0.shape)
    plt.contourf(xx0, xx1, yy)
    draw_scattrer(test_datas, plt)
    plt.show()


