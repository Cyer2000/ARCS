"""
辅助计算文件：PAD与节点的圆覆盖关系
"""
import tool_data as dl
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

points_file = "../data/number/single/nodes200-16000.txt"  # 测试用数据


class Dcircle(object):
    def __init__(self, id, center, covers, redundant):
        """
        覆盖圆类
        :param id: 唯一标识
        :param center: 圆心坐标
        :param nodes_id: 节点id集
        :param nodes_pos: 节点坐标集
        :param covers: 覆盖节点数
        :param redundant: 是否冗余
        """
        self.id = id
        self.center = center
        self.nodes_id = []
        self.nodes_pos = []
        self.covers = covers
        self.redundant = redundant


class node(object):
    def __init__(self, id, pos):
        """
        节点类
        :param id: 唯一标识
        :param pos: 位置信息
        """
        self.id = id
        self.pos = pos
        self.parents = []  # 这个节点所在的圆的id


def compute_cover(nodes, circles, radius):
    """
    计算每个初始圆的覆盖节点
    :param nodes: 节点集
    :param circles: 圆心集
    :return:
    """
    for i in nodes:
        i.parents = []
    for j in circles:
        j.nodes_pos = []
        j.nodes_id = []
        j.covers = 0
    for i in nodes:
        for j in circles:
            if dl.euler_distance(i.pos, j.center) <= radius:
                j.nodes_id.append(i.id)
                j.nodes_pos.append(i.pos)
                j.covers = j.covers + 1
                i.parents.append(j.id)


def remove_circle(circles, points_number, d_max):
    """
    移除多余覆盖圆
    :param circles: 圆心集
    :return:
    """

    circles_copy = circles.copy()
    # circles_copy.sort(key=lambda circles: len(circles.nodes_id), reverse=True)
    ifHaveRedundants = True
    while ifHaveRedundants:
        for i in circles_copy:
            circles_ = circles_copy.copy()
            circles_.remove(i)
            covered = check_cover(circles_)
            centers_ = []
            for j in circles_:
                centers_.append(j.center)
            ifConnect, _, disconnected = check_connect(centers_, d_max)
            if len(covered) == points_number and ifConnect:
                # BS不能被标记删除
                if circles.index(i) != 0:
                    i.redundant = True
                    circles_copy = circles_
                    break
            else:
                if i == circles_copy[-1]:
                    ifHaveRedundants = False
    new_circles = []
    new_centers = []
    delete_indexs = []
    for i in circles:
        if i.redundant == False:
            new_centers.append([round(i.center[0], 2), round(i.center[1], 2)])
            new_circles.append(i)
        else:
            delete_indexs.append(circles.index(i))
    delete_indexs.sort(reverse=True)
    return new_circles, new_centers, delete_indexs


def check_connect(centers, d_max):
    """
    检查当前圆集是否连通
    :param circles:
    :return:
    """
    G = nx.Graph()
    len_centers = len(centers)
    for i in range(len_centers):
        G.add_node(i)
    # 求出点集中每两个点的欧氏距离，因为是对称的所以循环的时候不用两个全循环
    for i in range(len_centers - 1):
        for j in range(i + 1, len_centers):
            a = round(dl.euler_distance(centers[i], centers[j]), 3)
            if round(dl.euler_distance(centers[i], centers[j]), 3) <= d_max:
                G.add_edge(i, j)
    # pos = nx.spring_layout(G, scale=5)
    # nx.draw(G, pos, node_size=1000, font_color='white', font_size=16, with_labels=True)
    # plt.show()
    disconnected = []
    connnected = []
    ifConnect = True
    for i in range(1, len(G.nodes())):
        # 非BS簇心到BS都有可达路径
        if nx.has_path(G, 0, i) is False:
            ifConnect = False
            disconnected.append(i)
        else:
            connnected.append(i)

    return ifConnect, connnected, disconnected


def check_cover(circles):
    """
    检查当前圆集中覆盖不重复节点数
    :param circles: 覆盖圆集合
    :return:
    """
    covered_id = []
    for i in circles:
        for j in i.nodes_id:
            if j not in covered_id:
                covered_id.append(j)
    return covered_id


def find_min_cover(circles, nodes, computed):
    """
    计算每个圆的单独覆盖节点数
    寻找单独覆盖节点数最小的圆（最可能被合并的圆）
    :param computed: 已经计算过的圆
    :return:
    """
    min_cover_circle = None
    min_covers = math.inf
    min_index = 0
    # 所有圆单独覆盖节点数集
    single_covers_count = [(0) for i in range(len(circles))]
    # 根据每个节点的父母计算每个圆的单独覆盖点数
    for i in nodes:
        if len(i.parents) == 1:
            single_covers_count[i.parents[0]] = single_covers_count[i.parents[0]] + 1
    # 寻找最少单独覆盖点数的圆
    # print("single_covers_count")
    # print(single_covers_count)
    for j in range(len(single_covers_count)):
        # 不能选择BS来漂移
        if j != 0 and (j not in computed) and single_covers_count[j] < min_covers:
            min_covers = single_covers_count[j]
            min_cover_circle = circles[j]
            min_index = j
    return min_cover_circle, min_index


def find_single_covers(circles, target_circles, nodes):
    """
    寻找目标圆集中单独覆盖的节点
    :param circles: 所有圆集
    :param target_circles:目标圆集
    :return:
    """
    target_covers = check_cover(target_circles)
    rest_circles = [x for x in circles if x not in target_circles]
    rest_covers = check_cover(rest_circles)
    target_single_covers = [y for y in target_covers if y not in rest_covers]
    target_single_covers_pos = list(map(lambda x: list(nodes[x].pos), target_single_covers))

    return target_single_covers_pos


def show_clusters(data, centers, radius):
    """
    节点覆盖示意
    :param data: 所有数据
    :param centers: 聚类中心
    :param radius: 类半径
    :return:
    """
    colors = dl.get_color_dict()
    plt.figure(figsize=(10, 8))
    plt.xlim((-100, 16100))
    plt.ylim((-100, 16100))
    # plt.scatter(X[:, 0], X[:, 1], s=20)
    theta = np.linspace(0, 2 * np.pi, 800)
    plt.scatter(data[:, 0], data[:, 1], color=colors[0], s=20)
    for i in range(len(centers)):
        centroid = centers[i]
        plt.scatter(centroid[0], centroid[1], color=colors[5], marker='x', s=100)
        x, y = np.cos(theta) * radius + centroid[0], np.sin(theta) * radius + centroid[1]
        plt.plot(x, y, linewidth=1, color=colors[2])
    plt.axis('scaled')
    plt.axis('equal')
    plt.show()


def get_circles(centers, point_list):
    """
    构造节点集和圆集
    :param centers: 圆心集
    :param point_list: 数据集
    :return:构造圆集和节点集
    """
    circles = []
    nodes = []
    for i in range(len(centers)):
        circles.append(Dcircle(i, centers[i], 0, False))
    for j in range(len(point_list)):
        nodes.append(node(j, point_list[j]))

    return circles, nodes


def ifHaveCover(center, point_list, r):
    """
    根据输入初始中心判断是否有点被中心圆覆盖
    :param center: 当前中心
    :param point_list: 节点集
    :return:
    """
    circles, nodes = get_circles(center, point_list)
    compute_cover(nodes, circles, r)
    center_cover_points = check_cover(circles)
    if len(center_cover_points) == 0:
        return False
    else:
        return True


def get_cover(center, point_list, r):
    """
    根据输入中心返回覆盖节点集
    :param center:中心
    :param point_list:所有节点
    :param r:半径
    :return:返回节点位置和节点索引
    """
    circles, nodes = get_circles(center, point_list)
    compute_cover(nodes, circles, r)
    covers = []
    indexes = []
    for i in circles:
        covers.extend(i.nodes_pos)
        indexes.extend(i.nodes_id)
    return covers, indexes


def delete(centers, point_list, radius, d_max):
    """
    根据输入圆心集和数据集删除冗余圆
    :param centers: 圆心集
    :param point_list: 数据集
    :param radius: 圆半径
    :return: 绘图显示删除冗余之后的结果并返回新圆心集
    """
    # print(len(centers))
    circles, nodes = get_circles(centers, point_list)
    compute_cover(nodes, circles, radius)
    circles, centers, delete_indexs = remove_circle(circles, len(point_list), d_max)
    # print(len(centers))
    # show_clusters(point_list, centers, radius)

    return centers, delete_indexs


def full_coverage(centers, point_list, radius):
    """
    根据输入圆心集和数据集判断目前是否全覆盖
    :param centers: 圆心集
    :param point_list: 数据集
    :param radius: 圆半径
    :return: 是否全覆盖
    """
    circles, nodes = get_circles(centers, point_list)
    compute_cover(nodes, circles, radius)
    uncovered_nodes = []
    covered_nodes = check_cover(circles)
    len_nodes = len(point_list)
    ifFullCoverage = False
    if len(covered_nodes) == len_nodes:
        ifFullCoverage = True
    else:
        for i in range(len_nodes):
            if i not in covered_nodes:
                uncovered_nodes.append(i)
    return ifFullCoverage, uncovered_nodes


def get_average(centers, point_list, radius):
    """
    根据输入圆心集和数据集获得覆盖点平均值
    :param centers:
    :param point_list:
    :param radius:
    :return:
    """
    circles, nodes = get_circles(centers, point_list)
    compute_cover(nodes, circles, radius)
    covered = check_cover(circles)
    sum = np.array([0, 0])
    for i in covered:
        sum = sum + np.array(nodes[i].pos)

    return (sum/len(covered)).tolist()


def get_targets_single_covers(centers, target_centers, point_list, radius):
    """
    根据目标圆心集计算其覆盖范围内的单独覆盖点
    :param centers: 所有圆心集
    :param target_centers: 目标圆心集
    :return: 单独覆盖点集
    """
    circles, nodes = get_circles(centers, point_list)
    compute_cover(nodes, circles, radius)
    target_circles = []
    for i in target_centers:
        target_circles.append(circles[i])
    target_pos = find_single_covers(circles, target_circles, nodes)
    return target_pos


def main():
    centers = []
    radius = 300
    point_list = dl. get_points(points_file)
    get_cover(centers, point_list, radius)


if __name__ == "__main__":
    main()


