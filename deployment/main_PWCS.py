import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.special import comb, perm
import networkx as nx
import copy
import sys
import csv
import os

import UAV as U
import tool_circle as cir
from tool_data import get_points, get_color_dict
from tool_calculate import (E_dis, check_BS, get_ouliers, find_nearest_from_two_set, move_center, find_nearest,
                            sort_nearest_tuple, find_furthest_tuple, find_furthest, outer_center)
from base_K_cover import KMeans
from base_tsp import ortools_tsp
import base_cvrp


class DEPLOY(object):
    def __init__(self, points_list, d_max, radius, BS, shift, pads):
        self.points = points_list
        self.d = d_max
        self.r = radius
        self.BS = BS
        self.BS_nodes = []
        self.centers = []
        self.shift = shift
        self.pads = pads

    def KOP_cluster(self, alpha):
        """
        KMeans聚类算法
        """
        self.BS_nodes, self.points = check_BS(self.BS, self.points, self.r)
        self.centers = get_ouliers(self.points, {}, alpha)
        # print('self.centers:', self.centers)
        if len(self.centers) != 0:
            o_KM = KMeans(len(self.centers))
            self.centers = o_KM.fit(np.array(self.points), np.array(self.centers))

    def coverage(self):
        """
        在未覆盖节点上添加PAD
        """
        new_pads = []
        _, uncovered_nodes = cir.full_coverage(self.centers, self.points, self.r)
        while uncovered_nodes:
            self.centers.append(self.points[uncovered_nodes[0]])
            new_pads.append(self.points[uncovered_nodes[0]])
            _, uncovered_nodes = cir.full_coverage(self.centers, self.points, self.r)
        return new_pads

    def check_constraints(self):
        """
        检查覆盖性
        :return:
        """
        ifCoverage, _ = cir.full_coverage(self.centers, self.points, self.r)
        return ifCoverage

    def detele(self):
        """
        删除冗余圆
        """
        # 调整一下centers的顺序
        new_centers = [self.BS]
        centers = self.centers.copy()
        while centers:
            con_center, next_center, dis = find_nearest_from_two_set(new_centers, centers)
            new_centers.append(next_center)
            centers.remove(next_center)
        self.centers = new_centers
        # 删除冗余圆
        new_centers = self.centers.copy()
        centers, del_indexs = cir.delete(new_centers, self.points, self.r, self.d)
        detele_centers = []
        for i in del_indexs:
            if self.centers[i] != self.BS:
                detele_centers.append(self.centers[i])
            del self.centers[i]
        return detele_centers

    def shift_cluster(self):
        """
        簇心漂移
        """
        # 簇心偏移
        circles, cir_nodes = cir.get_circles(self.centers, self.points)
        move_queue = []
        # 迭代停止条件：所有簇都已完成漂移
        while len(move_queue) < len(self.centers) - 1:
            # 计算当前圆集覆盖点数和单独覆盖点数
            cir.compute_cover(cir_nodes, circles, self.r)
            # 找到单独覆盖节点数最小的圆作为下一个漂移圆
            min_cover_circle, min_index = cir.find_min_cover(circles, cir_nodes, move_queue)
            # 在已漂移队列中加入该圆
            move_queue.append(min_index)
            select_center = min_cover_circle.center
            # print(select_center)
            # 找到当前漂移圆的最近簇心邻居（作为漂移目标）
            neighbor = find_nearest(select_center, self.centers)
            # 计算初始簇心与漂移目标的距离
            init_distance = E_dis(select_center, neighbor)
            move_distance = 0
            # 迭代停止条件：在漂移后不满足约束
            old_centers = []
            while self.check_constraints():
                # 保留漂移前元素
                old_centers = self.centers.copy()
                # 向最近邻居漂移shift距离
                new_center = move_center(select_center, neighbor, self.shift)
                # 更新簇心
                self.centers[min_cover_circle.id] = new_center
                select_center = new_center
                # 判断漂移距离是否超出漂移目标
                move_distance = move_distance + self.shift
                if move_distance > init_distance:
                    break
            self.centers = old_centers
            # 获取新圆心和点集
            circles, cir_nodes = cir.get_circles(self.centers, self.points)
        # 漂移后示意图

    def combine_cluster_circle(self):
        """
        利用外接圆合并簇
        """
        center_tuples = sort_nearest_tuple(self.centers, self.d)
        check_count = 0
        # 归并迭代
        while True:
            # 若整个过程都没有进入continue，说明已经找到新簇心可以合并
            if check_count == len(center_tuples):
                break
            # 寻找下一个最近簇心对
            closest_centers_id = center_tuples[check_count][0:2]
            # 基站不能动
            if 0 in closest_centers_id:
                # 尝试计数增加，进入下一次迭代
                check_count = check_count + 1
                continue

            centers = self.centers.copy()
            # 寻找单独覆盖的节点
            covered_points = cir.get_targets_single_covers(centers, closest_centers_id, self.points, self.r)
            if len(covered_points) < 2:
                check_count = check_count + 1
                continue
            # 寻找最远节点对，计算距离和终点
            fur_part, fur_part_dis, new_mid = find_furthest_tuple(covered_points)
            rest_covered_points = [x for x in covered_points if x not in fur_part]
            if fur_part_dis > self.r * 2:
                check_count = check_count + 1
                continue
            # 寻找距离中点最远的点
            fur_line_point = find_furthest(new_mid, rest_covered_points)
            closest_centers = list(map(lambda x: centers[x], closest_centers_id))
            self.centers = [x for x in centers if x not in closest_centers]
            self.centers.append(new_mid)
            if not self.check_constraints():
                # 若中点不能作为新簇心
                self.centers.pop()
                # 对单独覆盖节点数过少的特殊情况进行排除,直接检查下一PAD对
                if len(rest_covered_points) == 0:
                    self.centers = centers.copy()
                    check_count = check_count + 1
                    continue
                # 尝试用三角形外接圆合并
                else:
                    fur_part.append(fur_line_point)
                    # 计算三角形外接圆心
                    new_outer = outer_center(fur_part)
                    # new_outers.append(new_outer)
                    self.centers.append(new_outer)
                    if not self.check_constraints():
                        self.centers = centers.copy()
                        check_count = check_count + 1
                        continue
            # 两个旧簇心已经合并，需要置零尝试计数，重新检查簇心合并可能
            center_tuples = sort_nearest_tuple(self.centers, self.d)
            check_count = 0


def show_clusters(nodes, radius, BS, pads=None, covers=None, deletes=None, residences=None, mp_path=None, uav_path=None, fig_dir=None):
    """
    节点覆盖示意图
    :param nodes: 传感器节点集合
    :param radius: 充电点的覆盖半径
    :param BS: 基站坐标
    :param pads: 已经存在的固定PAD
    :param covers: 为满足覆盖性约束增加的充电点
    :param deletes: 冗余删去的充电点
    :param residences:
    :param mp_path:
    :param uav_path:
    :param fig_dir:
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

    # 若是满足覆盖性约束的充电点
    if covers is not None:
        for i in covers:
            plt.scatter(i[0] / 10000, i[1] / 10000, color=colors[1], marker='*', s=800)
            bs = Circle((i[0] / 10000, i[1] / 10000), radius, alpha=0.1, color='#44546A')
            ax.add_artist(bs)

    # 若是冗余待删除的充电点
    if deletes is not None:
        for i in covers:
            plt.scatter(i[0] / 10000, i[1] / 10000, color=colors[1], marker='*', s=800)
            bs = Circle((i[0] / 10000, i[1] / 10000), radius, alpha=0.1, color='#44546A')
            ax.add_artist(bs)

    # 若充电点是固定PAD
    if pads is not None:
        for i in range(len(pads)):
            centroid = pads[i]
            plt.scatter(centroid[0] / 10000, centroid[1] / 10000, color='#ED7D31', marker='^', s=600)
            c = Circle((centroid[0] / 10000, centroid[1] / 10000), radius, alpha=0.1, color='#7f00ff')
            ax.add_artist(c)

    # 如果UAV路径存在
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

    # 如果有停留点
    if residences is not None:
        for i in range(len(residences)):
            centroid = residences[i]
            c = Circle((centroid[0] / 10000, centroid[1] / 10000), 0.025, alpha=0.5, color='#70AD47')
            ax.add_artist(c)
            cover = Circle((centroid[0] / 10000, centroid[1] / 10000), radius, alpha=0.1, color='#70AD47')
            ax.add_artist(cover)

    # 最后画传感器节点
    plt.scatter(nodes[:, 0] / 10000, nodes[:, 1] / 10000, color='#1569C5', s=50, label='sensor node')

    # 图的坐标轴边界
    plt.axis('equal')
    plt.xticks([]) # 设置x坐标轴
    plt.yticks([]) # 设置y坐标轴
    ax = plt.gca()  # 获得坐标轴的句柄
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


def get_distant_nodes(nodes, charging_stations, radius):
    """
    找到预部署PADs和BS部署范围以外的节点
    """
    covers, indexes = cir.get_cover(charging_stations, nodes, radius)
    distant_nodes = nodes.copy()
    for i in range(len(covers)):
        for j in range(len(distant_nodes)):
            if (distant_nodes[j] == covers[i]).all():
                distant_nodes = np.delete(distant_nodes, j, axis=0)
                break
    return distant_nodes


def get_residence(distant_nodes, pads, d_max, radius, BS, shift, alpha):
    """
    对distant_nodes调用KOP_DSC算法获得移动PAD的停留点位置
    """
    data = distant_nodes.copy().tolist()
    o_deploy = DEPLOY(data, d_max, radius, BS, shift, pads)

    # 计算孤立节点的数量，Kmeans聚类作为初始移动PAD停留点
    # print("Start KOP-DSC...")
    o_deploy.KOP_cluster(alpha)
    centers = o_deploy.centers
    centers.append(BS)
    if len(centers) == 1:
        return centers
    # print("Initial kmeans number of centers:"+str(len(centers)))

    # 检查覆盖性
    cover_centers = o_deploy.coverage()
    centers = o_deploy.centers
    # print(len(centers), "centers:", centers)

    # 删除冗余PAD
    delete_centers = o_deploy.detele()
    centers = o_deploy.centers
    # print(len(centers), "delete centers:", centers)

    # 漂移PAD
    o_deploy.shift_cluster()
    centers = o_deploy.centers
    # print(len(centers), "shift centers:", centers)

    # 合并PAD
    o_deploy.combine_cluster_circle()
    centers = o_deploy.centers
    # print(len(centers), "combine centers:", centers)

    return centers


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


def complete_residences(residences, unlimited_big_circle, d_max, BS):
    """
    为了满足UAV能量的连通性约束，加入新的residences得到big_circle
    """
    big_circle_tmp = []
    mp_path_tmp = []

    for i in range(len(unlimited_big_circle) - 1):
        # 加入集合
        big_circle_tmp.append(unlimited_big_circle[i])
        if unlimited_big_circle[i] in residences:
            mp_path_tmp.append(unlimited_big_circle[i])
        # 若不满足能量的连通性
        if E_dis(unlimited_big_circle[i], unlimited_big_circle[i + 1]) > d_max:
            new_point = move_center(unlimited_big_circle[i], unlimited_big_circle[i + 1], d_max)
            big_circle_tmp.append(new_point)
            residences.append(new_point)
            mp_path_tmp.append(new_point)
            while E_dis(new_point, unlimited_big_circle[i + 1]) > d_max:
                new_point = move_center(new_point, unlimited_big_circle[i + 1], d_max)
                big_circle_tmp.append(new_point)
                residences.append(new_point)
                mp_path_tmp.append(new_point)

    big_circle = big_circle_tmp
    mp_path = mp_path_tmp

    # 最后加上BS
    big_circle.append(BS)
    mp_path.append(BS)

    return residences, big_circle, mp_path


def get_uav_path(big_circle, nodes, radius, d_max):
    """
    得到UAV的包含big_circle和所有small_circles的节点充电路径
    """
    # 初始化
    uav_path = []

    # 得到所有charging stations(cs)范围内覆盖的节点
    covers, indexes = get_cs_covers(big_circle, nodes, radius)

    # 沿着big_circle，在charging station处运行vrp
    for i in range(len(big_circle) - 1):

        # 求即将要运行YRP的vrp_points：该charging station + near_nodes
        vrp_points = [big_circle[i]]
        # 将充电点圆形覆盖的节点加入vrp_points
        for j in range(len(covers[i])):
            vrp_points.append(np.array(covers[i][j]))

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

    # 最后加上BS
    uav_path.append(big_circle[0])

    return uav_path


def get_time(uav_path, mp_cycle, BS, pads, uav_vel, mp_vel, charge_time, charged_time):
    """
    得到UAV和MP移动的总时间
    """
    uav_time = 0
    mp_time = 0
    mp_start = BS

    for i in range(len(uav_path) - 1):
        # UAV的起点和终点
        uav_start = uav_path[i]
        uav_end = uav_path[i + 1]
        uav_distance = E_dis(uav_start, uav_end)
        # uav_time加上路上的时间
        uav_time = uav_time + uav_distance / uav_vel
        # 若UAV遇到了充电点（固定PAD或BS），uav_time加上被充电的时间
        if (uav_end == BS or uav_end in pads) and i != len(uav_path) - 2:
            uav_time = uav_time + charged_time
        # 若UAV遇到了和移动PAD的相遇点，uav_time加上被充电的时间，并比较更新UAV和移动PAD的时间
        elif uav_end in mp_cycle and i != len(uav_path) - 2:
            # mp的终点是目前的目标点
            mp_end = uav_end
            mp_distance = E_dis(mp_start, mp_end)
            mp_time = mp_time + mp_distance / mp_vel
            # 若UAV先到，更新uav_time与mp一致
            if mp_time >= uav_time:
                mp_time = mp_time + charged_time
                uav_time = mp_time
            # 若mp先到，更新mp_time与UAV一致
            else:
                uav_time = uav_time + charged_time
                mp_time = uav_time
            # 更新mp新的起点
            mp_start = mp_end
        # 若遇到的是普通节点，uav_time加上给节点充电的时间
        else:
            uav_time = uav_time + charge_time
        # 若UAV回到终点，uav_time不变，mp_time要增加
        if i == len(uav_path) - 2:
            # mp的终点是目前的目标点
            mp_end = uav_end
            mp_distance = E_dis(mp_start, mp_end)
            mp_time = mp_time + mp_distance / mp_vel
        # print('uav_time:', uav_time, 'mp_time:', mp_time)

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
    # 计算UAV的参数
    o_uav = U.UAV(uav_vel, uav_max_E, BS)
    o_uav.computePower()
    radius = round(o_uav.maxRadius(node_E), 3)  # 无人机充电半径
    d_max = round(o_uav.maxDistance(), 3)  # 无人机最大飞行距离
    P_mov = o_uav.P_mov  # 无人机推进功率
    P_hov = o_uav.P_hov  # 无人机悬停功率
    shift = 30  # 漂移距离
    alpha = 0.3  # 控制k的数量

    # 画初始图
    # show_clusters(nodes, radius, BS, pads)

    # ————————————————————————————第一步：Nodes Division————————————————————————————

    # 得到charging stations：预部署PADs与BS
    charging_stations = [BS]
    for i in pads:
        charging_stations.append(i)
    # print('charging_stations:', charging_stations)

    # 得到distant_nodes：charging stations（预部署PADs与BS）覆盖范围之外的节点
    distant_nodes = get_distant_nodes(nodes, charging_stations, radius)
    # print('distant_nodes:', distant_nodes)

    # ————————————————————————————第二步：CDC&DSC————————————————————————————

    # 调用CDC&DSC得到residences：移动PAD的停留点，包括BS
    residences = get_residence(distant_nodes, pads, d_max, radius, BS, shift, alpha)
    # print('residences:', residences)

    # 更新charging_stations = pads + residences(其中第一个元素就为BS)
    charging_stations = residences.copy()
    for i in range(len(pads)):
        charging_stations.append(pads[i])
    print('charging_stations:', charging_stations)

    # ————————————————————————————第三步：Node Charging Path Determination————————————————————————————

    # 得到unlimited_big_circle：不满足能量限制的在charging stations之间的大循环
    unlimited_big_circle = tsp(charging_stations)
    # print('energy_unlimited_big_circle:', unlimited_big_circle)

    # 得到big_circle：满足能量限制的在charging stations之间的大循环；以及mp_path
    residences, big_circle, mp_path = complete_residences(residences, unlimited_big_circle, d_max, BS)
    # print('big_circle:', big_circle)
    # print('mp_path:', mp_path)
    # show_clusters(nodes, radius, BS, pads, [], [], residences, [], big_circle)
    # show_clusters(nodes, radius, BS, pads, [], [], residences, mp_path)
    # show_clusters(nodes, radius, BS, pads, [], [], residences, mp_path, big_circle)

    # 得到uav_path：big_circle加上所有small_circles
    uav_path = get_uav_path(big_circle, nodes, radius, d_max)
    # print('uav_path:', uav_path)
    # show_clusters(nodes, radius, BS, pads, [], [], residences, [], uav_path)

    # ————————————————————————————计算时间、导出图片等————————————————————————————

    # 计算MP和UAV移动的总时间
    uav_time, mp_time = get_time(uav_path, mp_path, BS, pads, uav_vel, mp_vel, charge_time, charged_time)
    # print('uav_time:', uav_time, 'mp_time:', mp_time)

    # 计算uav消耗的总能量
    uav_consumption, uav_charging_efficiency = count_uav_consumption(uav_path, len(nodes), node_E, uav_vel, P_mov, P_hov)

    # 返回结果
    print('uav_time:', uav_time, '; uav_consumption:', uav_consumption, '; uav_charging_efficiency:', uav_charging_efficiency)
    return uav_time, uav_consumption, uav_charging_efficiency


def test():
    """
    这个文件的测试入口
    """
    # 变量的默认值：网络条件、节点数量、区域大小、pads数量、uav携带能量、uav速度
    BS = [8000, 8000]  # 基站坐标
    nodes_file = 'E:\Lab\mobile-pad\\2410\data\\number\\100~1000-16000-10\\100\\0nodes100-16000.txt'  # 节点文件
    pads = [[8000, 3000], [13000, 13000], [8000, 13000]]  # 固定pads
    uav_max_E = 60000  # 无人机电池容量
    uav_vel = 20  # 无人机飞行速度
    # 测试运行
    main(BS, nodes_file, pads, uav_max_E, uav_vel)

if __name__ == '__main__':
    # 测试方法
    test()