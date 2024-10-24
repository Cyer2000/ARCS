import math
import numpy as np
import tool_data as dl
import tool_circle as cir


def E_dis(point1: np.ndarray, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return round(math.sqrt(distance), 3)


def check_BS(BS, points_list, r):
    """
    检查BS覆盖节点并移除这些节点
    :param BS: BS位置
    :return: 返回覆盖节点和新的points_list
    """
    covers, indexes = cir.get_cover([BS], points_list, r)
    indexes.sort(reverse=True)
    for i in indexes:
        points_list.pop(i)
    return covers, points_list


def get_ouliers(points_list, distances, alpha):
    """
    计算离散节点确定K值
    :param nodes: 节点集
    :param distances: 距离
    :return:离散节点集
    """
    neighbors = []
    outliers = []
    # 计算每个节点的最近节点
    for i in range(len(points_list)):
        min_dist = math.inf  # 定义浮点正无限常量
        min_neighbor = None
        for j in range(len(points_list)):
            # 为了不重复计算距离，保存在字典内
            if i != j:
                d_key = (i, j)
                if d_key not in distances:
                    distances[d_key] = E_dis(points_list[i], points_list[j])
                d = distances[d_key]
                if d < min_dist:
                    min_neighbor = points_list[j]
                    min_dist = d
        neighbors.append([points_list[i], min_neighbor, min_dist])
    # 计算距离期望E
    neighbors.sort(key=lambda x: x[-1], reverse=True)
    Sum = 0
    for i in neighbors:
        Sum = i[-1] + Sum
    dis_E = Sum / len(neighbors)
    # 若距离大于期望值则被判定成孤立节点
    for i in range(len(neighbors)):
        if neighbors[i][-1] > dis_E:
            outliers.append(neighbors[i][0])
    K = int(len(outliers) * alpha)
    outliers = outliers[0:K]
    # print('out:', len(outliers))

    return outliers


def find_nearest_from_two_set(set1, set2):
    """
    寻找两个集合中最近的对
    :param set1:集合1
    :param set2:集合2
    :return:
    """
    nodes_dis = math.inf
    closeset_part = None
    for i in set1:
        for j in set2:
            if E_dis(i, j) < nodes_dis:
                closeset_part = (i, j)
                nodes_dis = E_dis(i, j)
    center1, center2 = closeset_part
    return center1, center2, nodes_dis


def move_center(departure, arrival, shift):
    """
    簇心向最近簇漂移一段指定距离
    :param departure: 出发簇心
    :param arrival: 漂移方向簇心
    :param shift: 漂移距离
    :return: 新的簇心位置
    """
    distance = dl.euler_distance(departure, arrival)
    new_center = []
    for a, b in zip(departure, arrival):
        new_center.append(a + (((b - a) * shift) / (distance)))
    for i in range(len(new_center)):
        new_center[i] = new_center[i]

    return new_center


def find_nearest(center, centers):
    """
    寻找距离指定簇心最近的簇心
    :param center: 指定簇心
    :return: 最近簇心
    """
    min_distance = math.inf
    target = None
    for i in centers:
        if center != i and E_dis(i, center) < min_distance:
            target = i
            min_distance = dl.euler_distance(i, center)
    return target


def sort_nearest_tuple(centers, d):
    """
    将簇心对按照最近距离排序
    :param centers:簇心集
    :return:
    """
    # 节点对
    centers_tuple = []
    for i in range(len(centers) - 1):
        for j in range(i + 1, len(centers)):
            curr_distance = E_dis(centers[i], centers[j])
            if curr_distance <= d:
                centers_tuple.append([i, j, curr_distance])
    centers_tuple.sort(key=lambda x: x[-1])
    return centers_tuple


def find_furthest_tuple(points):
    """
    在节点集中找最远的两个节点
    :param points: 节点集
    :return: 最远节点对, 相距距离
    """
    points_len = len(points)
    max_dist = 0
    fur_tuple = None
    for i in range(points_len - 1):
        for j in range(i + 1, points_len):
            distance = E_dis(points[i], points[j])
            if distance > max_dist:
                fur_tuple = [points[i], points[j]]
                max_dist = distance
    mid = []
    for a, b in zip(fur_tuple[0], fur_tuple[1]):
        mid.append((a + b) / 2)
    return fur_tuple, max_dist, mid


def find_furthest(center, points):
    """
    寻找距离指定簇心最远的点
    :param center: 指定簇心
    :param points: 节点集
    :return: 最远点
    """
    max_distance = 0
    target = None
    for i in points:
        if E_dis(i, center) > max_distance:
            target = i
            max_distance = E_dis(i, center)
    return target


def outer_center(points):
    """
    根据3个点组成的三角形寻找外接圆心
    :param points: 3个点的集合
    :return: 外心坐标
    """
    x1 = points[0][0]
    x2 = points[1][0]
    x3 = points[2][0]
    y1 = points[0][1]
    y2 = points[1][1]
    y3 = points[2][1]
    A1 = 2 * (x2 - x1)
    B1 = 2 * (y2 - y1)
    C1 = x2 * x2 - x1 * x1 - y1 * y1 + y2 * y2
    A2 = 2 * (x3 - x2)
    B2 = 2 * (y3 - y2)
    C2 = x3 * x3 - x2 * x2 - y2 * y2 + y3 * y3
    x = ((C1 * B2) - (C2 * B1)) / ((A1 * B2) - (A2 * B1))
    y = ((A1 * C2) - (A2 * C1)) / ((A1 * B2) - (A2 * B1))

    return [x, y]
