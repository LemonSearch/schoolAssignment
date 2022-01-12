import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def kmeans(X, k, r, max_step):
    # 随机生成k个聚类中心
    center = X[np.random.randint(low=len(X), size=k)]

    # y_pre是预测的类别，初始化类别
    y_pre = np.zeros((X.shape[0], 1))

    # 迭代寻找好的聚类中心
    step = 0
    move = np.inf
    while step < max_step and move > r:
        move = 0
        # 用聚类中心给样本分类
        for i in range(X.shape[0]):
            dist = np.linalg.norm(X[i] - center[int(y_pre[i])])
            for j in range(k):
                new_dist = np.linalg.norm(X[i] - center[j])
                if dist > new_dist:
                    dist = new_dist
                    y_pre[i] = j

        # 根据分类重新确定聚类中心
        for i in range(k):
            old = center[i].copy()
            center[i, 0] = np.average(X[np.where(y_pre == i)[0], 0])
            center[i, 1] = np.average(X[np.where(y_pre == i)[0], 1])
            move += np.linalg.norm(old - center[i])
        step = step + 1

    # 迭代相关信息
    print("迭代步数：" + str(step))
    print("最后一次迭代的移动距离：" + str(move))

    # 根据最新聚类中心再分类一次
    for i in range(X.shape[0]):
        dist = np.linalg.norm(X[i] - center[int(y_pre[i])])
        for j in range(k):
            new_dist = np.linalg.norm(X[i] - center[j])
            if dist > new_dist:
                dist = new_dist
                y_pre[i] = j

    return y_pre, center


# 读取数据文件
# 处理数据，把标签（质量）分成两类，用0-1变量表示
# 将标签和特征值分类存放
dataset = load_iris()
X = dataset["data"]
y = dataset["target"]
y = y.reshape(y.size, 1)
dataset = np.concatenate((X, y), axis=1)
dataset = pd.DataFrame(dataset)
dataset = dataset[dataset[4] != 2]
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# 数据预处理
# 对数据进行PCA降维
pca = PCA(n_components=2)
X = pca.fit_transform(X)
# 对X特征数据进行归一化
sc = MinMaxScaler()
X = sc.fit_transform(X)

# k-means聚类的k值
k = int(input("请输入k值："))

# 绘图
plt.figure(figsize=(5, 5))
ax = plt.gca()
y_pre, center = kmeans(X, k, 0.000001, 1000)
print("各聚类中心的坐标：")
print(center)
plt.scatter(X[:, 0], X[:, 1], c=y_pre, s=10, cmap=plt.cm.Paired, label="样本集")
plt.scatter(center[:, 0], center[:, 1], c='k', s=50, marker='x', label="聚类中心")
plt.title("聚类结果（k="+str(k)+"）")  # 图表标题
ax.legend()  # 添加图例

# 设置正常显示中文
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.tight_layout()
plt.show()
