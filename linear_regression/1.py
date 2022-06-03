import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(filename):
    """
    函数说明：加载数据
    :param filename: 文件名
    :return: xArr - x数据集  yArr - y数据集
    """
    numFeat = len(open(filename).readline().split()) - 1

    xArr = []
    yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.split()
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def plotDataSet():
    """
    函数说明：绘制数据集
    :return:
    """
    xArr, yArr = loadDataSet('ex0.txt')
    n = len(xArr)  # 数据个数
    xcord = [];
    ycord = []
    for i in range(n):
        print(i)
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


def standRegress(xArr, yArr):
    """
    函数说明：计算回归系数w
    :param xArr: x数据集
    :param yArr: y数据集
    :return: ws：回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # 判断是否为可逆
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def plotRegression():
    """
    函数说明：绘制回归曲线和数据点
    :return:
    """
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegress(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    xCopy.sort(0)  # 排序
    yHat = xCopy * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:, 1], yHat, c='red')
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plotRegression()

