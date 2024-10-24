"""
辅助计算文件：数据处理
"""
import random
import numpy as np
import os
import math


color_dict = {
    0:'#000000', # 黑色
    1:'#0000FF', # 蓝色
    2:'#009900', # 绿色
    3:'#EE0000', # 红色
    4:'#FFCC00', # 黄色
    5:'#009966', # 黑色
    6:'#99CC00', # 黑色
    7:'#996666', # 黑色
    8:'#FF33FF', # 黑色
    9:'#FF9999', # 黑色
    10:'#CCFFFF' # 黑色
}


def get_color_dict():
    """
    分类做图标签
    :return: 颜色字典
    """
    return color_dict


def get_random(low, high):
    """
    随机生成指定区间随机数
    :param low: 下限
    :param high: 上限
    :return:
    """
    return((high-low)*random.random()+low)


def generate(points_number, area_size, filename):
    """
    随机生成节点
    :param points_number: 节点数目
    :param area_size: 区域范围大小
    :param filename: 存入文件名
    """
    n = 0
    with open(filename, 'w') as f:
        while n < points_number:
            rand_tuple = []
            x = round(get_random(0, area_size), 3)
            y = round(get_random(0, area_size), 3)
            f.write(str(x) + ',' + str(y) + '\n')
            rand_tuple.append(x)
            rand_tuple.append(y)
            n += 1
        if n == points_number:
            print("After", n, "tries,can't get a random point!Check whether the problem has a solution!")


def get_points(filename):
    """
    读取数据
    :param filename: 读取文件名
    :return: 节点列表
    """
    points_list = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = [float(x) for x in line.split(',')]
            points_list.append(line)
    points_list = np.array(points_list)
    return points_list


def euler_distance(point1, point2):
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)

    return math.sqrt(distance)


def w_list(rlist, f):
    """
    写入一维列表
    """
    for y in rlist:
        f.write(str(y) + '  ')
    f.write("\n")


def w_list_3D(rlist, f):
    """
    写入三维列表(PAD数量结果)
    :param: 三维列表
    """
    for line in rlist:
        for x in line:
            for y in range(len(x)):
                if y < len(x) - 1:
                    f.write(str(x[y]) + ',')
                else:
                    f.write(str(x[y]))
            f.write('  ')
        f.write("\n")


def r_list_3D(f, type, read_lines, read_context):
    """
    读出三维列表(PAD数量结果)
    :param type: 转化类型
    :param read_lines:阅读行数
    :return:
    """
    D3_list = []
    ifStartRead = False
    while read_lines != 0:
        per_list = []
        if ifStartRead is not True:
            line = f.readline()
            if line == read_context:
                ifStartRead = True
        else:
            per_val = []
            line = f.readline().strip('\n').split()
            for i in line:
                i_ = i.split(",")
                if type == "f":
                    per_val = list(map(lambda x: float(x), i_))
                if type == "i":
                    per_val = [float(i_[0]), int(i_[1])]
                per_list.append(per_val)
            D3_list.append(per_list)
            read_lines = read_lines - 1

    return D3_list


def r_list(f, type):
    """
    读出一行列表
    :param type:转化类型
    :return:
    """
    if type == "f":
        rlist = list(map(lambda x: float(x), f.readline().strip('\n').split()))
    if type == 'i':
        rlist = list(map(lambda x: int(x), f.readline().strip('\n').split()))
    return rlist


def list_separation(olist):
    """
    拆分二维列表
    :param olist:
    :return:
    """
    x_list = []
    y_list = []
    for i in olist:
        x_list.append(i[0])
        y_list.append(i[1])
    return x_list, y_list


def get_filenames(file_dir):
    """
    获得文件夹下所有目录名
    :param file_dir: 目录名
    :return:
    """
    list_path = []
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        # if os.path.isdir(file_path):
        #     os.listdir(file_path, list_name)
        # else:
        list_path.append(file_path)
    return list_path


def get_number_filenames(file_dir):
    """
    获得文件夹下int类型的目录名
    :param file_dir:
    :return:
    """
    list_name = []
    list_path = []
    for file in os.listdir(file_dir):
        list_name.append(int(file))
    list_name.sort()
    for i in list_name:
        file_path = os.path.join(file_dir, str(i))
        list_path.append(file_path)
    return list_path


def main():
    a = [9257.29692300709, 8653.898331532844]
    b = [5312.178511963627, 5924.354631750686]
    c = euler_distance(a, b)


if __name__ == '__main__':
    main()



