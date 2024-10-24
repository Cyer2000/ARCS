"""
利用谷歌Ortools进行TSP求解
"""
import math
import numpy as np
from matplotlib import pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def compute_euclidean_distance_matrix(locations):
    """
    计算距离矩阵
    :param locations: 节点集合
    :return:
    """
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances


def create_data_model(distance, num_vehicle, start):
    """
    将数据处理成ortools的数据类型
    :param distance: 距离矩阵
    :param num_vehicle: 车的数量
    :param start: 起点索引
    :return:
    """
    data = {}
    data['distance_matrix'] = distance
    data['num_vehicles'] = 1  # 问题中的车辆数量，TSP时为1
    data['depot'] = 0  # 起点索引
    return data


def ortools_tsp(tsp_points):
    """
    程序的入口点
    :param tsp_points: 需要实现tsp的点集合
    :return:
    """
    # 将点化成距离矩阵
    distance = compute_euclidean_distance_matrix(tsp_points)
    # print(distance)

    # 实例化问题数据
    data = create_data_model(distance, 1, 0)

    # 创建路由索引管理器
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # 创建路由模型
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        # 返回两个节点之间的距离, 将路由变量索引转换为距离矩阵节点索引
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 定义每个边的成本
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 设置第一个解决方案启发式
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # 求解问题
    solution = routing.SolveWithParameters(search_parameters)

    # 返回求解结果
    ordered_points = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        ordered_points.append(tsp_points[index])
        previous_index = index
        index = solution.Value(routing.NextVar(index))
    ordered_points.append(tsp_points[routing.Start(0)])
    route_distance = solution.ObjectiveValue()

    return ordered_points, route_distance


# 文件测试
def main():
    tsp_points = [[988.683, 11962.282], [2924.932, 13319.357], [3564.449, 9068.503], [2351.098, 12632.4], [1185.179, 13033.168], [3503.798, 7293.392], [1680.566, 12162.982], [3033.073, 1134.25], [1828.63, 13893.653], [1632.51, 8682.768], [2590.274, 9591.505], [100.924, 5883.785], [2861.745, 12989.076], [3650.087, 13649.246], [143.154, 3787.595], [3622.557, 5926.26], [3173.809, 8966.754], [1931.215, 5314.596], [3732.862, 9992.517], [1341.553, 347.562], [3167.348, 4238.918], [2080.186, 6424.642], [1126.489, 6370.405], [487.246, 15353.173], [2360.214, 10437.21], [3798.419, 13254.092], [2827.043, 14440.409], [2144.312, 3332.955], [2583.966, 15611.642], [3421.715, 15160.869], [918.673, 3960.185], [2547.227, 11732.704], [821.766, 10956.527], [2117.818, 193.114], [2754.531, 4238.826], [2986.307, 1170.363], [2164.971, 5706.841], [1252.044, 4825.471], [84.03, 462.305], [1573.356, 2031.482], [2138.544, 10247.651], [1743.243, 8683.94], [2847.778, 11599.943], [1458.157, 2896.888], [1795.22, 6411.57], [545.918, 12730.198], [2846.594, 1307.64], [304.568, 456.943], [1619.788, 11852.837], [2467.619, 11477.247], [2436.518, 6217.773], [2325.993, 10599.499], [3375.437, 3563.21], [1733.0, 12097.61], [3359.181, 15237.289], [3275.992, 9349.856], [3273.373, 5993.765], [2657.713, 8579.861], [1565.917, 14114.451], [2286.823, 1219.799], [2985.25, 14083.872]]
    tsp_points = np.array(tsp_points)

    # 画初始图
    fig = plt.figure(figsize=(2, 10))
    plt.scatter(tsp_points[:, 0] / 10000, tsp_points[:, 1] / 10000, color='#1569C5', s=20, label='sensor node')
    plt.scatter(tsp_points[0, 0] / 10000, tsp_points[0, 1] / 10000, color='#ff00ff', s=50, label='sensor node')
    plt.show()

    # 运行ortools_tsp
    ordered_points, route_distance = ortools_tsp(tsp_points)
    ordered_points = np.array(ordered_points)
    print(ordered_points)
    print(route_distance)

    # 画最后的路线图
    fig = plt.figure(figsize=(2, 10))
    plt.scatter(ordered_points[:, 0] / 10000, ordered_points[:, 1] / 10000, color='#1569C5', s=20, label='sensor node')
    plt.plot(ordered_points[:, 0] / 10000, ordered_points[:, 1] / 10000, color='#ff00ff', label='sensor node')
    plt.show()


if __name__ == '__main__':
    main()