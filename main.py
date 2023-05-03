import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # 随机森林,GBDT
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import sklearn
import copy

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 设置正负号

#手搓代码岭回归
def dataNumalize(data):
    """
    归一化
    param data:传入的数据
    return:标准化之后的数据
    """
    new_data = (data - data.min()) / (data.max() - data.min())
    return new_data

def DataPreProcessing(data):
    """
       对性别序列二值化
       param data:传入的数据
       return:标准化之后的数据
       """
    #性别序列处理
    if data.ndim > 1:
        for i in range(data.shape[0]):
            if data[i][1] > 0:
                data[i][1] = 1
            else:
                data[i][1] = 0
    #-----------------------------------------#
    new_data = (data - data.min()) / (data.max() - data.min())
    return new_data

def lossFuncation(x, y, theta):  # 输入是numpy不是pandas
    """
       平方损失函数改进的经验风险
       param data:传入的数据和迭代权重系数
       return:损失值
       """
    n = y.shape[0]
    data = (np.dot(x, theta) - y) ** 2
    loss = data.sum() / (2 * n)
    return loss


def BGD(x, y, theta, alpha, lamba, num_iters):
    """
       岭回归，随机梯度下降
       param data:固定步长为alpha，lambda为正则项的权重，theta为单次迭代后的权重系数，num_iters为迭代次数。
       return:[theta，num_iters]
       """
    m = y.shape[0]  # m是指训练集的个数
    loss_all = []  # 负责存入每次更新的loss存入
    theta_all = []
    num_iter = 0
    loss_min = 10000
    for i in range(num_iters):
        f = np.dot(x.T, np.dot(x, theta).reshape(m, ) - y) / m
        theta = theta - alpha * f.reshape(5, 1)
        theta[:-1] = theta[:-1] - alpha * (lamba / m) * theta[:-1]  # 考虑到bias，lamba可以不除以m
        loss = lossFuncation(x, y, theta)
        loss_all.append(loss)
        theta_all.append(theta)
        if loss_min > loss:
            num_iter = i + 1
            loss_min = loss
            theta_min = theta_all[i]
        #print("第{}次的loss值为{}".format((i + 1), loss))  # 完成题目打印每次的损失值
    return theta_min, loss_all, num_iter, theta_all,loss_min


# 数据预处理
diabetes = datasets.load_diabetes()
data = diabetes['data'][:, 0:4]
target = diabetes['target']
#data = dataNumalize(data)
data = DataPreProcessing(data)
target = dataNumalize(target)
feature_names = diabetes['feature_names'][0:4]
df = pd.DataFrame(data, columns=feature_names)
# 这里用的是常规的分数据集方法4：1其实也可以改后面的0.8的参数，这里没试过，主要数据集数量不是影响的主要因素
train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(data, target, train_size=0.8)

#岭回归
train_X_lasso = np.insert(train_X, 4, 1, axis=1)  # 插入了bias
test_X_lasso = np.insert(test_X, 4, 1, axis=1)
theta_lasso = np.ones((train_X_lasso.shape[1], 1))  # 初始化权重为1，这里可以用for循环更简单
alpha = 0.01  # 初始化步长，可以改
lamba = 0.05  # 同样可以改
num_iters = 1000  # 迭代次数
theta_lasso, loss_all, num_iter, theta_all, loss_min = BGD(train_X_lasso, train_Y, theta_lasso, alpha, lamba, num_iters)
print("lasso训练集loss最小为第{}次\n对应的theta值为\n{}".format(num_iter, theta_lasso))
y_pred_lasso = np.dot(test_X_lasso, theta_lasso)  # 预测的值
loss_lasso = lossFuncation(test_X_lasso, test_Y, theta_lasso)
print("岭回归训练集loss为{}\n岭回归测试集loss为{}".format(loss_min, loss_lasso))
print("岭回归平均绝对误差：", mean_absolute_error(test_Y, y_pred_lasso))
print("岭回归均方误差：", mean_squared_error(test_Y, y_pred_lasso))
print('岭回归r2(决定系数): ', r2_score(test_Y, y_pred_lasso))

#手敲随机森林（没时间，暂时先不敲了）

model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
model.fit(train_X, train_Y)
y_pred_rfr = model.predict(test_X)

print("\n随机森林回归平均绝对误差：", mean_absolute_error(test_Y, y_pred_rfr))
print("随机森林回归均方误差：", mean_squared_error(test_Y, y_pred_rfr))
print('随机森林回归r2(决定系数): ', r2_score(test_Y, y_pred_rfr))

#GradientBoostingRegressor

model_gbr = GradientBoostingRegressor(n_estimators=100)
model_gbr.fit(train_X, train_Y)
y_pred_gbdt = model_gbr.predict(test_X)

print("\nGBDT回归平均绝对误差：", mean_absolute_error(test_Y, y_pred_gbdt))
print("GBDT回归均方误差：", mean_squared_error(test_Y, y_pred_gbdt))
print('GBDT回归r2(决定系数): ', r2_score(test_Y, y_pred_gbdt))




plt.figure()
plt.title(u"关于糖尿病病因预测结果")
plt.scatter(np.arange(len(test_Y)), test_Y[:], color='red', marker="3", label='真实label')  # 真实label
plt.scatter(np.arange(len(test_Y)), y_pred_lasso[:], color='skyblue', marker="8", label='岭回归预测的label')  # 预测的label
plt.scatter(np.arange(len(test_Y)), y_pred_rfr[:], color='orange', marker="4", label='随机森林预测的label')
plt.scatter(np.arange(len(test_Y)), y_pred_gbdt[:], color='green', marker="3", label='GBDT预测的label')
plt.legend(loc='upper right')  # 设置图例的位置
plt.show()
