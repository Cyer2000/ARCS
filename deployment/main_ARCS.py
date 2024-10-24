import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import UAV as U
import tool_circle as cir
from tool_data import get_points, get_color_dict
from tool_calculate import E_dis, move_center
from base_tsp import ortools_tsp
import base_cvrp


def show_clusters(nodes, radius, BS, pads=None, rendezvous=None, mp_path=None, uav_path=None, fig_dir=None):
    """
    画图函数
    :param nodes: 所有数据
    :param radius: 半径
    :param BS: 基站坐标
    :param pads: 已经存在的固定PADs
    :param rendezvous: 移动PAD的会合点
    :param mp_path: 移动PAD行进的路径
    :param uav_path: uav行进的路径
    :param fig_dir: 图片保存的地址
    :return:展示图
    """

    # 初始化参数
    radius = radius / 10000
    colors = get_color_dict()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    theta = np.linspace(0, 2 * np.pi, 800)

    # 若是基站
    plt.scatter(BS[0] / 10000, BS[1] / 10000, color=colors[1], marker='*', s=800)
    bs = Circle((BS[0] / 10000, BS[1] / 10000), radius, alpha=0.1, color='#7f00ff')
    ax.add_artist(bs)

    # 若充电点是固定PADs
    if pads is not None:
        for i in range(len(pads)):
            centroid = pads[i]
            plt.scatter(centroid[0] / 10000, centroid[1] / 10000, color='#ED7D31', marker='^', s=600)
            c = Circle((centroid[0] / 10000, centroid[1] / 10000), radius, alpha=0.1, color='#7f00ff')
            ax.add_artist(c)

    # 如果uav路径存在
    if uav_path is not None:
        for i in range(len(uav_path) - 1):
            start = uav_path[i]
            end = uav_path[i + 1]
            x, y = [start[0] / 10000, end[0] / 10000], [start[1] / 10000, end[1] / 10000]
            plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], length_includes_head=True, linewidth=4, color='#F5CF44',
                      head_width=0.02, head_length=0.02)

    # 如果移动PAD路径存在
    if mp_path is not None:
        for i in range(len(mp_path) - 1):
            start = mp_path[i]
            end = mp_path[i + 1]
            x, y = [start[0] / 10000, end[0] / 10000], [start[1] / 10000, end[1] / 10000]
            plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], alpha=1, linestyle=':', length_includes_head=True, linewidth=2.5, color='#70AD47',
                      head_width=0.02, head_length=0.05)

    # 如果有会合点
    if rendezvous is not None:
        for i in range(len(rendezvous)):
            centroid = rendezvous[i]
            c = Circle((centroid[0] / 10000, centroid[1] / 10000), 0.025, alpha=0.5, color='#70AD47')
            ax.add_artist(c)

    # 最后画传感器节点
    plt.scatter(nodes[:, 0] / 10000, nodes[:, 1] / 10000, color='#1569C5', s=50, label='sensor node')

    # 图的坐标轴边界
    plt.axis('equal')
    plt.xticks([])  # 设置x坐标轴
    plt.yticks([])  # 设置y坐标轴
    ax.spines['bottom'].set_linewidth(1.2)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.2)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.2)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.2)  # 设置上部坐标轴的粗细

    # 如果有地址的话，保存图片
    if fig_dir is not None:
        plt.savefig(fig_dir)
        plt.close()
    else:
        plt.show()  # 展示图片


def get_uncovered_nodes(nodes, charging_stations, radius):
    """
    找到固定PADs和BS部署范围以外的节点
    """
    covers, indexes = cir.get_cover(charging_stations, nodes, radius)
    uncovered_nodes = nodes.copy()
    for i in range(len(covers)):
        for j in range(len(uncovered_nodes)):
            if (uncovered_nodes[j] == covers[i]).all():
                uncovered_nodes = np.delete(uncovered_nodes, j, axis=0)
                break
    return uncovered_nodes


def get_tsp_points(uncovered_nodes, BS, pads):
    """
    得到TSP需要经过的点：uncovered_nodes、BS、pads
    """
    tsp_points = [BS]
    for node in uncovered_nodes:
        tsp_points.append([node[0], node[1]])
    for pad in pads:
        tsp_points.append([pad[0], pad[1]])
    return tsp_points


def tsp(points):
    """
    运行tsp
    """
    big_circle, distance = ortools_tsp(points)
    return big_circle


def get_cs_covers(centers, nodes, radius):
    """
    得到每个charging station覆盖的near-nodes
    """
    # 初始化数组
    covers = []
    indexes = []
    for i in range(len(centers)):
        covers.append([])
        indexes.append([])
    # 遍历每个传感器节点
    for i in range(len(nodes)):
        min_dis = 100000
        min_j = -1
        for j in range(len(centers)):
            dis = E_dis(nodes[i], centers[j])
            if dis < min_dis:
                min_dis = dis
                min_j = j
        if min_dis < radius:
            covers[min_j].append(nodes[i])
            indexes[min_j].append(i)
    return covers, indexes


def get_uav_path(big_circle, charging_stations, covers, d_max):
    """
    得到uav的包含big_circle和所有small_circles的节点充电路径
    """
    # 初始化
    uav_path = []
    cs_endpoints = [[] for i in range(len(charging_stations))]  # 每个charging station的最后一个出发节点

    # 沿着big_circle，在charging station处运行vrp
    for i in range(len(big_circle) - 1):

        # 若遇到cs：先vrp经过范围内除end_node的节点，再前往end_node然后去下一个节点
        if big_circle[i] in charging_stations:

            # 求即将要运行VRP的vrp_points：该charging station + covered_nodes - end_node 的集合
            vrp_points = [big_circle[i]]
            end_node = []  # “end_node”定义为此cs中与下一个节点最近的covered_node
            min_d = math.inf  # 暂时的最短距离
            next_node = big_circle[i + 1]  # 下一个节点
            curr_cs_index = 0  # 当前charging station的序号
            for j in range(len(charging_stations)):
                if big_circle[i] == charging_stations[j]:
                    curr_cs_index = j
                    for k in range(len(covers[j])):
                        if E_dis(covers[j][k], next_node) < min_d:
                            min_d = E_dis(covers[j][k], next_node)
                            end_node = covers[j][k]
                    # 把不是end_node的其余范围内covered_nodes加入vrp_points
                    for k in range(len(covers[j])):
                        if not (covers[j][k] == end_node).all():
                            vrp_points.append(covers[j][k])

            # 对vrp_points运行vrp
            demand = [0 for j in range(len(vrp_points))]
            # 若工作范围内无节点则distance=0
            if len(vrp_points) == 1:
                distance = 0
                small_circle = []
            else:
                distance, small_circle = base_cvrp.main(vrp_points, demand, 200, d_max)

            # 将small_circle加入uav_path
            for j in range(len(small_circle)):
                for k in range(len(small_circle[j]) - 1):
                    uav_path.append(np.array(vrp_points[small_circle[j][k]]).tolist())
            uav_path.append(np.array(vrp_points[0]).tolist())

            # 将end_node加入uav_path
            if np.array(end_node).any():
                uav_path.append(np.array(end_node).tolist())
                cs_endpoints[curr_cs_index] = np.array(end_node).tolist()

        else:
            uav_path.append(big_circle[i])

    # 最后加上BS
    uav_path.append(big_circle[0])

    return uav_path, cs_endpoints


def update_big_circle(big_circle, charging_stations, cs_endpoints):
    """
    得到加上cs_endpoints的big_circle
    """
    new_big_circle = []

    # 去掉BS循环big_circle，沿路加上cs_endpoints
    for i in range(len(big_circle) - 1):
        new_big_circle.append(big_circle[i])
        for j in range(len(charging_stations)):
            if big_circle[i] == charging_stations[j] and cs_endpoints[j]:
                new_big_circle.append(cs_endpoints[j])
    # 加上BS
    new_big_circle.append(big_circle[0])

    return new_big_circle


def get_rendezvous(uav_path, charging_stations, new_big_circle, uav_max_E, P_mov, P_hov, uav_vel, charge_time, node_E):
    """
    沿着uav_path得到uav与移动PAD的会合点，以及加上会合点的uav_path_rendezvous
    """
    rendezvous = []
    uav_path_rendezvous = []
    curr_E = uav_max_E

    for i in range(len(uav_path) - 1):

        # 加入uav_path_rendezvous
        uav_path_rendezvous.append(uav_path[i])

        # 开始和结束节点
        start = uav_path[i]
        end = uav_path[i + 1]

        # 若开始节点不属于new_big_circle，即为covered_nodes，则跳过
        if start not in new_big_circle:
            continue

        # 若开始节点是charging stations，则电量为满电
        if start in charging_stations:
            curr_E = uav_max_E

        # 计算路径中的使用能量
        distance = E_dis(start, end)
        E_mov = P_mov * distance / uav_vel
        E_hov = P_hov * charge_time
        # 计算到end的剩余能量，若下一个是充电点/若下一个是node
        if end in charging_stations:
            tmp_E = curr_E - E_mov
        else:
            tmp_E = curr_E - E_mov - E_hov - node_E

        # 如果能量不够，增加会合点
        new_point = start
        while tmp_E <= 0:
            # 求会合点位置
            remain_d = curr_E / P_mov * uav_vel
            new_point = move_center(new_point, end, remain_d)
            # 将会合点加入rendezvous集合和uav_path_rendezvous集合
            rendezvous.append(new_point)
            uav_path_rendezvous.append(new_point)
            curr_E = uav_max_E
            # 计算路径中的使用能量
            distance = E_dis(new_point, end)
            E_mov = P_mov * distance / uav_vel
            E_hov = P_hov * charge_time
            # 计算剩余能量，若下一个是充电点/若下一个是node
            if end in charging_stations:
                tmp_E = curr_E - E_mov
            else:
                tmp_E = curr_E - E_mov - E_hov - node_E
        # 更新curr_E
        curr_E = tmp_E

    # 加上BS
    uav_path_rendezvous.append(uav_path[0])

    return rendezvous, uav_path_rendezvous


def count_time(uav_path_rendezvous, mp_path, BS, pads, uav_vel, mp_vel, charge_time, charged_time):
    """
    得到uav和mp移动的总时间(charging delay)
    """
    uav_time = 0
    mp_time = 0
    mp_start = mp_path[0]

    for i in range(len(uav_path_rendezvous) - 1):
        # uav的起点和终点
        uav_start = uav_path_rendezvous[i]
        uav_end = uav_path_rendezvous[i + 1]
        uav_distance = E_dis(uav_start, uav_end)
        # uav_time加上路上的时间
        uav_time = uav_time + uav_distance / uav_vel
        # 若uav遇到了charging stations（固定PADs或BS），uav_time加上被充电的时间
        if (uav_end == BS or uav_end in pads) and i != len(uav_path_rendezvous) - 2:
            uav_time = uav_time + charged_time
        # 若uav遇到了和移动PAD的相遇点，uav_time加上被充电的时间，并比较更新uav和移动PAD的时间
        elif uav_end in mp_path and i != len(uav_path_rendezvous) - 2:
            # mp的终点是目前的目标点
            mp_end = uav_end
            mp_distance = E_dis(mp_start, mp_end)
            mp_time = mp_time + mp_distance / mp_vel
            # 若uav先到，更新uav_time与mp_time一致
            if mp_time >= uav_time:
                mp_time = mp_time + charged_time
                uav_time = mp_time
            # 若mp先到，更新mp_time与uav_time一致
            else:
                uav_time = uav_time + charged_time
                mp_time = uav_time
            # 更新mp新的起点
            mp_start = mp_end
        # 若遇到的是普通节点，uav_time加上给节点充电的时间
        else:
            uav_time = uav_time + charge_time

    # uav_time要减去最后在BS充电的时间
    uav_time = uav_time - charged_time

    return uav_time, mp_time

def count_uav_consumption(uav_path_rendezvous, num_nodes, node_E, uav_vel, P_mov, P_hov):
    """
    得到uav的总消耗和充电效率
    """
    consumption = 0
    # 路径上的推进能耗
    for i in range(len(uav_path_rendezvous) - 1):
        consumption += E_dis(uav_path_rendezvous[i], uav_path_rendezvous[i + 1]) / uav_vel * P_mov
    # 为节点充电的能耗
    charging_consumption = (P_hov + node_E) * num_nodes
    consumption += charging_consumption
    # 计算充电效率：充电能耗占所有能耗
    charging_efficiency = charging_consumption / consumption
    # 返回总消耗和充电效率
    return consumption, charging_efficiency

def main(BS, nodes_file, pads, uav_max_E, uav_vel):

    # 设置参数
    nodes = get_points(nodes_file)  # 节点
    node_E = 30  # 节点最大携带能量
    mp_vel = 10  # 移动PAD移动速度
    charge_time = 2  # 给任意一个传感器节点充电的时间
    charged_time = round(uav_max_E / 87.42, 2)  # uav被充电的时间
    # uav相关参数
    o_uav = U.UAV(uav_vel, uav_max_E, BS)
    o_uav.computePower()
    radius = round(o_uav.maxRadius(node_E), 3)  # 无人机充电半径
    d_max = round(o_uav.maxDistance(), 3)  # 无人机最大飞行距离
    P_mov = o_uav.P_mov  # 无人机推进功率
    P_hov = o_uav.P_hov  # 无人机悬停功率

    # 画初始图
    # show_clusters(nodes, radius, BS, pads)

    # ————————————————————————————第一步：Nodes Division————————————————————————————

    # 得到charging stations：固定PADs与BS
    charging_stations = [BS]
    for i in pads:
        charging_stations.append(i)
    # print('charging_stations:', charging_stations)

    # 得到uncovered_nodes：charging stations（固定PADs与BS）覆盖范围之外的节点
    uncovered_nodes = get_uncovered_nodes(nodes, charging_stations, radius)
    # print('uncovered_nodes:', uncovered_nodes)
    if len(uncovered_nodes) == 0:
        print("uncovered_nodes's number is 0")
        return

    # ————————————————————————————第二步：Node Charging Path Determination————————————————————————————

    # 得到tsp_points，即uav大循环要经过的点：uncovered_nodes、charging stations（BS和固定PADs）
    tsp_points = get_tsp_points(uncovered_nodes, BS, pads)
    # print('tsp_points:', tsp_points)

    # TSP求解得到big_circle：uav大循环按顺序经过的点（从BS出发回到BS）
    big_circle = tsp(tsp_points)
    # print('big_circle:', big_circle)
    # show_clusters(nodes, radius, BS, pads, [], [], big_circle)

    # 得到每个charging station覆盖的covered_node
    covers, indexes = get_cs_covers(charging_stations, nodes, radius)

    # 得到uav的节点充电路径
    uav_path, cs_endpoints = get_uav_path(big_circle, charging_stations, covers, d_max)
    # print("uav_path:", uav_path)
    # print("cs_endpoints:", cs_endpoints)
    # show_clusters(nodes, radius, BS, pads, [], [], uav_path)

    # ————————————————————————————第三步：Rendezvous Scheduling————————————————————————————

    # 更新big_circle：加上cs_endpoints
    new_big_circle = update_big_circle(big_circle, charging_stations, cs_endpoints)
    # print("new_big_circle:", new_big_circle)

    # 得到所有移动PAD的会合点
    rendezvous, uav_path_rendezvous = get_rendezvous(uav_path, charging_stations, new_big_circle,
                                                     uav_max_E, P_mov, P_hov, uav_vel, charge_time, node_E)
    # print('rendezvous:', rendezvous)
    # print('uav_path_rendezvous:', uav_path_rendezvous)

    # 得到mp的路径
    mp_path = rendezvous.copy()
    mp_path.append(rendezvous[0])  # 回到起点而非基站，节省时间

    # ————————————————————————————画图、计算时间、导出图片等————————————————————————————

    # 画图
    show_clusters(nodes, radius, BS, pads, rendezvous, [], uav_path)
    # show_clusters(nodes, radius, BS, pads, [], mp_path, [])
    # show_clusters(nodes, radius, BS, pads, [], [], uav_path_rendezvous)

    # 计算mp和uav移动的总时间
    uav_time, mp_time = count_time(uav_path_rendezvous, mp_path, BS, pads, uav_vel, mp_vel, charge_time, charged_time)
    print('uav_time:', uav_time, 'mp_time:', mp_time)

    # 计算uav消耗的总能量
    uav_consumption, uav_charging_efficiency = count_uav_consumption(uav_path_rendezvous, len(nodes), node_E, uav_vel, P_mov, P_hov)

    # 返回结果
    # print('uav_time:', uav_time, '; uav_consumption:', uav_consumption, '; uav_charging_efficiency:', uav_charging_efficiency)
    return uav_time, uav_consumption, uav_charging_efficiency

def test():
    """
    这个文件的测试入口
    """
    # 变量的默认值：网络条件、节点数量、区域大小、pads数量、uav携带能量、uav速度
    BS = [8000, 8000]  # 基站坐标
    # nodes_file = '../../charge_uav/data/nodes/nodes160-16000.txt'  # 节点文件
    nodes_file = 'E:\Lab\mobile-pad\\2410\data\\number\\100~1000-16000-10\\100\\0nodes100-16000.txt'  # 节点文件
    pads_file = 'E:\Lab\mobile-pad\\2410\data\\pad_num\\padNodesLocation-8.txt'  # pad位置文件
    # pads = [[8000, 3000], [13000, 13000], [8000, 13000]]  # 固定pads
    pads = get_points(pads_file).tolist()
    print('pads:', pads)
    uav_max_E = 50000  # 无人机电池容量
    uav_vel = 20  # 无人机飞行速度
    # 测试运行
    main(BS, nodes_file, pads, uav_max_E, uav_vel)

if __name__ == '__main__':
    # 测试方法
    test()
