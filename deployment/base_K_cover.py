"""
K-means聚类（指定输入初始点）
"""
import math
import numpy as np
from tool_data import get_points, get_color_dict, generate
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.special import comb, perm
import networkx as nx
import copy
import tool_data as dl
import tool_circle as cir
# from sklearn import datasets
# from sklearn import cluster

area_size = 16000
points_number = 100
# 聚类簇的数量
n_cluster = 6


class KMeans(object):
    def __init__(self, k = 1):
        # print("k:", k)
        assert k > 0
        self.k = k
        self.labels = None

    def fit(self, points, centers):
        n = len(points)
        dis = np.zeros([n, self.k + 1])  # 各个点到类中心距离的距离矩阵
        self.labels = [-1] * n
        # 1.选择初始聚类中心
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            # 2.求各个点到k个聚类中心距离
            for i in range(n):
                curr = points[i]
                for j in range(self.k):
                    center = centers[j]
                    dis[i, j] = np.sqrt((curr[0] - center[0])**2 + (curr[1] - center[1])**2)  # 第i个点到第j个中心的距离
                # 3.归类
                # 距离矩阵每个点所在行最后一列为类编号
                dis[i, self.k] = np.argmin(dis[i, :self.k])  # 将值较小的下标值赋值给dis[i, 2]
                self.labels[i] = int(dis[i, self.k])

            # 4.求新的聚类中心
            centers_new = []
            point_x = points[:, 0]
            point_y = points[:, 1]
            for v in range(self.k):
                index = dis[:, self.k] == v
                if len(point_x[index]) > 0:
                    center_new = np.array([point_x[index].mean(), point_y[index].mean()])
                else:
                    center_new = centers[v]

                centers_new.append(center_new)
            # 5.判定聚类中心是否发生变换
            for q in range(len(centers)):
                if (centers[q] == centers_new[q]).all():
                    clusterChanged = False
                    # 如果都没发生变换则退出循环，表示已得到最终的聚类中心
                else:
                    clusterChanged = True
                    break

            centers = centers_new
        for i in range(len(centers)):
            curr = centers[i].tolist()
            centers.pop(i)
            curr = [round(x, 3) for x in curr]
            centers.insert(i, curr)
        return centers

    def calc_cluster(self, x):
        cluster = [([]) for i in range(self.k)] # 类样本点集
        for i in range(len(x)):
            k = self.labels[i] # 获得该样本点类别号
            cluster[k].append(list(x[i])) # 根据类别号向cluster中添加样本点

        return cluster


def k_means(data):

    my = KMeans(k1)
    centers = my.fit(data)
    new_centers = []
    for i in centers:
        i = i.tolist()
        j = []
        for x in i:
            j.append(round(x, 3))
        new_centers.append(j)
    C = my.calc_cluster(data)
    for i in range(len(C)):
        C[i] = np.array(C[i])

    return C, new_centers


def main():
    # x = np.linspace(0, 99, 100)
    # y = np.linspace(0, 99, 100)

    point_list = get_points(points_file)
    point_list = np.array(point_list)
    my = KMeans(k1)
    centers = my.fit(point_list)
    new_centers = []
    for i in centers:
        i = i.tolist()
        j = []
        for x in i:
            j.append(round(x, 3))
        new_centers.append(j)
    # print(new_centers)
    color_dict = get_color_dict()
    C = my.calc_cluster(point_list)

    for i in range(len(C)):
        C[i] = np.array(C[i])
        color = str(color_dict[i])
        plt.scatter(C[i][:, 0], C[i][:, 1], c=color, marker='o')
    # print(C)

    plt.show()


if __name__ == '__main__':
    main()
