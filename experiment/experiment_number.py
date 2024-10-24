"""
综合实验文件
"""
import datetime
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Ellipse, Circle

sys.path.append("../deployment/")
import tool_data as dl
import main_ARCS
import main_PWCS

def get_position(bs_type, pad_num, area_size):
    """
    计算BS和PAD的初始位置
    :param bs_type: BS的坐标类型
    :param pad_num: PAD的数量
    :param area_size: 区域大小
    :return: BS, pads
    """
    BS = []
    pads = []

    if bs_type == 'CENTER':
        BS = [round(area_size / 2, 3), round(area_size / 2, 3)]
    if bs_type == 'ISOLATED_IN':
        BS = [0, 0]

    if pad_num == 1:
        pads.append([round(area_size * 3 / 4, 3), round(area_size * 3 / 4, 3)])
    if pad_num == 2:
        pads.append([round(area_size * 3 / 4, 3), round(area_size * 3 / 4, 3)])
        pads.append([round(area_size / 4, 3), round(area_size / 4, 3)])
    if pad_num == 3:
        pads.append([round(area_size * 4 / 5, 3), round(area_size * 3 / 4, 3)])
        pads.append([round(area_size / 5, 3), round(area_size / 4, 3)])
        pads.append([round(area_size * 3 / 5, 3), round(area_size / 3, 3)])
    if pad_num == 4:
        pads.append([round(area_size * 3 / 4, 3), round(area_size * 3 / 4, 3)])
        pads.append([round(area_size / 4, 3), round(area_size / 4, 3)])
        pads.append([round(area_size * 3 / 4, 3), round(area_size / 4, 3)])
        pads.append([round(area_size / 4, 3), round(area_size * 3 / 4, 3)])
    if pad_num == 5:
        pads.append([round(area_size * 3 / 4, 3), round(area_size * 3 / 4, 3)])
        pads.append([round(area_size / 4, 3), round(area_size / 4, 3)])
        pads.append([round(area_size * 3 / 4, 3), round(area_size / 4, 3)])
        pads.append([round(area_size / 4, 3), round(area_size * 3 / 4, 3)])
        pads.append([round(area_size / 2, 3), round(area_size * 4 / 5, 3)])

    return BS, pads

def average_experiment(file_names, BS, pads, uav_max_E, uav_vel, variable, ARCS_res, PWCS_res):
    """
    进行一组10次平均实验
    """
    # 三个指标和平均实验相关参数
    sum_charge_delay = 0
    sum_uav_consumption = 0
    sum_charging_efficiency = 0
    index = 0

    # --------------此处进行ARCS实验----------------
    print("--start ARCS experiment--")
    for nodes_file in file_names:
        # print("This is the " + str(index) + " group of data.")
        charge_delay, uav_consumption, charging_efficiency = main_ARCS.main(BS, nodes_file, pads, uav_max_E, uav_vel)
        # 计算每组的实验数据
        sum_charge_delay += charge_delay
        sum_uav_consumption += uav_consumption
        sum_charging_efficiency += charging_efficiency
        index += 1
    # 求十组之后的平均值
    charge_delay_res = round(sum_charge_delay / int(index), 3)
    uav_consumption_res = round(sum_uav_consumption / int(index), 3)
    charging_efficiency_res = round(sum_charging_efficiency / int(index), 3)
    print("charge_delay: " + str(charge_delay_res) +
          " uav_consumption: " + str(uav_consumption_res) +
          " charging_efficiency: " + str(charging_efficiency_res))
    ARCS_res.append([variable, charge_delay_res, uav_consumption_res, charging_efficiency_res])

    # 重置
    sum_charge_delay = 0
    sum_uav_consumption = 0
    sum_charging_efficiency = 0
    index = 0

    # --------------此处进行PWCS实验----------------
    print("--start PWCS experiment--")
    for nodes_file in file_names:
        # print("This is the " + str(index) + " group of data.")
        charge_delay, uav_consumption, charging_efficiency = main_PWCS.main(BS, nodes_file, pads, uav_max_E, uav_vel)
        # 计算每组的实验数据
        sum_charge_delay += charge_delay
        sum_uav_consumption += uav_consumption
        sum_charging_efficiency += charging_efficiency
        index += 1
    # 求十组之后的平均值
    charge_delay_res = round(sum_charge_delay / int(index), 3)
    uav_consumption_res = round(sum_uav_consumption / int(index), 3)
    charging_efficiency_res = round(sum_charging_efficiency / int(index), 3)
    print("charge_delay: " + str(charge_delay_res) +
          " uav_consumption: " + str(uav_consumption_res) +
          " charging_efficiency: " + str(charging_efficiency_res))
    PWCS_res.append([variable, charge_delay_res, uav_consumption_res, charging_efficiency_res])

    return ARCS_res, PWCS_res

def output_log(log_dir, save_file, top_write, parameters, result, result_type):
    """
    实验结果输出log日志
    """
    # 在文件名的前面加上日期，然后和log_dir一起组装成最终保存文件的路径
    str_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')
    save_file = str_time + save_file
    save_path = os.path.join(log_dir, save_file)
    # 生成日志文件
    with open(save_path, "w") as f:
        f.write(top_write)
        f.write("Experiment time: " + str_time + "\n")
        f.write("Experiment parameters: " + "\n")
        f.write(parameters)
        f.write("Experiment result: " + "\n")
        for i in range(len(result)):
            f.write(result_type[i])
            dl.w_list(result[i], f)

def output_xlsx(results_ARCS, results_PWCS, xlsx_dir):
    """
    将数据输出到excel文件中
    """
    # 根据不同网络环境将数据拆分，一个网络环境的数据放在一个sheet下
    sheet_names = ['NORMAL-CENTER', 'NORMAL-ISOLATED_IN', 'GMM-CENTER', 'GMM-ISOLATED_IN']
    # 循环放不同网络环境的数据到不同sheet里
    for i in range(len(sheet_names)):
        data = [results_ARCS[i], results_PWCS[i]]
        # 创建一个DataFrame格式的数据
        df = pd.DataFrame(data)
        # 使用openpyxl引擎写入新工作表
        with pd.ExcelWriter(xlsx_dir, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheet_names[i], index=False)
    return

def experiment_number(data_file_dir, node_type, bs_type, pad_num, uav_vel, uav_max_E, log_dir):
    """
    实验1:节点密度变化
    """
    # 列出文件下所有数据合集的目录名列表
    all_dir = dl.get_filenames(data_file_dir)
    # 判断节点类型选择数据文件夹
    if node_type == 'NORMAL':
        multi_dir = all_dir[0]
    if node_type == 'GMM':
        multi_dir = all_dir[1]
    # 得到一些参数写在log里
    area_size = int(multi_dir.split("-")[-2])  # 区域大小：16000
    file_list = dl.get_number_filenames(multi_dir)  # 通过文件名得到不同节点数量的数据合集目录名列表(xx\\100, ... xx\\1000)
    MIN = int(file_list[0].split("\\")[-1])  # 变量的最小值和最大值
    MAX = int(file_list[-1].split("\\")[-1])
    # 判断BS位置类型和pad_num来设置BS坐标和pads坐标
    BS, pads = get_position(bs_type, pad_num, area_size)

    # ---对不同的节点数进行实验---
    print("-----Variable：Number-----")
    ARCS_res = []
    PWCS_res = []
    for i in file_list:
        # 得到此次变量的值，即从MIN到MAX的某个数字
        number = int(i.split("\\")[-1])
        print("the current number of nodes is " + str(number))
        # ---得到变量文件下的文件，进行一组10次平均实验---
        file_names = dl.get_filenames(i)
        ARCS_res, PWCS_res = average_experiment(file_names, BS, pads, uav_max_E, uav_vel, number, ARCS_res, PWCS_res)
    # 打印结果
    print('ARCS_res', ARCS_res)
    print('PWCS_res', PWCS_res)

    # ---输出文字日志---
    save_file = '-' + 'number' + '-' + str(node_type) + '-' + str(bs_type) + '.txt'
    top_write = "Variable index: The number of SNs(" + str(MIN) + "-" + str(MAX) + ")" + "\n"
    parameters = "The type of SNs: " + str(node_type) + "\n" \
                 + "The position of BS: " + str(bs_type) + "\n" \
                 + "The number of SNs: " + "[change]" + "\n" \
                 + "The number of pads: " + str(3) + "\n" \
                 + "The Region size: " + str(area_size) + "\n" \
                 + "The max energy of UAV: " + str(uav_max_E) + "\n" \
                 + "The velocity of UAV: " + str(uav_vel) + "\n"
    output_log(log_dir, save_file, top_write, parameters,[ARCS_res, PWCS_res], ['ARCS_res', 'PWCS_res'])

    return ARCS_res, PWCS_res

def experiment_size(data_file_dir, node_type, bs_type, pad_num, uav_vel, uav_max_E, log_dir):
    """
    实验2:监控区域变化
    :return:
    """
    # 列出文件下所有数据合集的目录名列表
    all_dir = dl.get_filenames(data_file_dir)
    # 判断节点类型选择数据文件夹
    if node_type == 'NORMAL':
        multi_dir = all_dir[0]
    if node_type == 'GMM':
        multi_dir = all_dir[1]
    # 得到一些参数写在log里
    number = int(multi_dir.split("-")[-2])  # 区域大小
    file_list = dl.get_number_filenames(multi_dir)  # 通过文件名得到不同节点数量的数据合集目录名列表(xx\\10000, ... xx\\25000)
    MIN = int(file_list[0].split("\\")[-1])  # 变量的最小值和最大值
    MAX = int(file_list[-1].split("\\")[-1])

    # ---对不同的区域大小进行实验---
    print("-----Variable：Size-----")
    ARCS_res = []
    PWCS_res = []
    for i in file_list:
        # 得到此次变量的值，即从MIN到MAX的某个数字
        area_size = int(i.split("\\")[-1])
        print("the current size of area is " + str(area_size))
        # 判断BS位置类型和pad_num来设置BS坐标和pads坐标
        BS, pads = get_position(bs_type, pad_num, area_size)
        # 得到变量文件下的10个文件，进行平均实验
        file_name = dl.get_filenames(i)
        ARCS_res, PWCS_res = average_experiment(file_name, BS, pads, uav_max_E, uav_vel, area_size, ARCS_res, PWCS_res)    # 打印结果
    print('ARCS_res', ARCS_res)
    print('PWCS_res', PWCS_res)

    # 输出文字日志
    save_file = '-' + 'size' + '-' + str(node_type) + '-' + str(bs_type) + '.txt'
    top_write = "Variable index: Region size(" + str(MIN / 10000) + "-" + str(MAX / 10000) + "  10^4 m^2)" + "\n"
    parameters = "The type of SNs: " + str(node_type) + "\n" \
                 + "The position of BS: " + str(bs_type) + "\n" \
                 + "The number of SNs: " + str(number) + "\n" \
                 + "The number of pads: " + str(3) + "\n" \
                 + "The Region size: " + "[change]" + "\n" \
                 + "The max energy of UAV: " + str(uav_max_E) + "\n" \
                 + "The velocity of UAV: " + str(uav_vel) + "\n"
    output_log(log_dir, save_file, top_write, parameters, [ARCS_res, PWCS_res], ['ARCS_res', 'PWCS_res'])

    return ARCS_res, PWCS_res

def experiment_pad(data_file_dir, node_type, bs_type, pad_num, uav_vel, uav_max_E, log_dir):
    """
    实验3:PAD数量变化
    """
    # 列出文件下所有数据合集的目录名列表
    all_dir = dl.get_filenames(data_file_dir)
    # 判断节点类型选择数据文件夹
    if node_type == 'NORMAL':
        multi_dir = all_dir[0]
    if node_type == 'GMM':
        multi_dir = all_dir[1]
    # 得到一些参数写在log里
    number = int(multi_dir.split("-")[-3])  # 节点数量：300
    area_size = int(multi_dir.split("-")[-2])  # 区域大小：16000
    file_names = dl.get_filenames(multi_dir)  # 通过文件名得到数据合集目录名列表
    MIN = 0  # 变量的最小值和最大值
    MAX = 4

    # ---对不同的PAD数进行实验---
    print("-----Variable：PAD-----")
    ARCS_res = []
    PWCS_res = []
    for i in range(0, 5):
        # 设置BS和pads
        pad_num = i
        BS, pads = get_position(bs_type, pad_num, area_size)
        # ---得到变量文件下的文件，进行一组10次平均实验---
        print("the current number of pads is " + str(pad_num))
        ARCS_res, PWCS_res = average_experiment(file_names, BS, pads, uav_max_E, uav_vel, pad_num, ARCS_res, PWCS_res)
    # 打印结果
    print('ARCS_res', ARCS_res)
    print('PWCS_res', PWCS_res)

    # ---输出文字日志---
    save_file = '-' + 'pad' + '-' + str(node_type) + '-' + str(bs_type) + '.txt'
    top_write = "Variable index: The number of pads(" + str(MIN) + "-" + str(MAX) + ")" + "\n"
    parameters = "The type of SNs: " + str(node_type) + "\n" \
                 + "The position of BS: " + str(bs_type) + "\n" \
                 + "The number of SNs: " + str(number) + "\n" \
                 + "The number of pads: " + "[change]" + "\n" \
                 + "The Region size: " + str(area_size) + "\n" \
                 + "The max energy of UAV: " + str(uav_max_E) + "\n" \
                 + "The velocity of UAV: " + str(uav_vel) + "\n"
    output_log(log_dir, save_file, top_write, parameters, [ARCS_res, PWCS_res], ['ARCS_res', 'PWCS_res'])

    return ARCS_res, PWCS_res

def experiment_energy(data_file_dir, node_type, bs_type, pad_num, uav_vel, uav_max_E, log_dir):
    """
    实验4:UAV能量变化
    """
    # 列出文件下所有数据合集的目录名列表
    all_dir = dl.get_filenames(data_file_dir)
    # 判断节点类型选择数据文件夹
    if node_type == 'NORMAL':
        multi_dir = all_dir[0]
    if node_type == 'GMM':
        multi_dir = all_dir[1]
    # 得到一些参数写在log里
    number = int(multi_dir.split("-")[-3])  # 节点数量：300
    area_size = int(multi_dir.split("-")[-2])  # 区域大小：16000
    file_names = dl.get_filenames(multi_dir)  # 通过文件名得到数据合集目录名列表
    MIN = 50000  # 变量的最小值和最大值
    MAX = 80000
    uav_max_E = MIN
    up_range = 5000

    # 判断BS位置类型和pad_num来设置BS坐标和pads坐标
    BS, pads = get_position(bs_type, pad_num, area_size)

    # ---对不同的UAV能量进行实验---
    print("-----Variable：Energy-----")
    ARCS_res = []
    PWCS_res = []
    while uav_max_E <= MAX:
        # ---得到变量文件下的文件，进行一组10次平均实验---
        print("the max energy of UAV is " + str(uav_max_E))
        ARCS_res, PWCS_res = average_experiment(file_names, BS, pads, uav_max_E, uav_vel, uav_max_E, ARCS_res, PWCS_res)
        uav_max_E = uav_max_E + up_range
    # 打印结果
    print('ARCS_res', ARCS_res)
    print('PWCS_res', PWCS_res)

    # ---输出文字日志---
    save_file = '-' + 'energy' + '-' + str(node_type) + '-' + str(bs_type) + '.txt'
    top_write = "Variable index: The max energy of UAV(" + str(MIN) + "-" + str(MAX) + ")" + "\n"
    parameters = "The type of SNs: " + str(node_type) + "\n" \
                 + "The position of BS: " + str(bs_type) + "\n" \
                 + "The number of SNs: " + str(number) + "\n" \
                 + "The number of pads: " + str(3) + "\n" \
                 + "The Region size: " + str(area_size) + "\n" \
                 + "The max energy of UAV: " + "[change]" + "\n" \
                 + "The velocity of UAV: " + str(uav_vel) + "\n"
    output_log(log_dir, save_file, top_write, parameters, [ARCS_res, PWCS_res], ['ARCS_res', 'PWCS_res'])

    return ARCS_res, PWCS_res

def experiment_velocity(data_file_dir, node_type, bs_type, pad_num, uav_vel, uav_max_E, log_dir):
    """
    实验4:UAV速度变化
    """
    # 列出文件下所有数据合集的目录名列表
    all_dir = dl.get_filenames(data_file_dir)
    # 判断节点类型选择数据文件夹
    if node_type == 'NORMAL':
        multi_dir = all_dir[0]
    if node_type == 'GMM':
        multi_dir = all_dir[1]
    # 得到一些参数写在log里
    number = int(multi_dir.split("-")[-3])  # 节点数量：300
    area_size = int(multi_dir.split("-")[-2])  # 区域大小：16000
    file_names = dl.get_filenames(multi_dir)  # 通过文件名得到数据合集目录名列表
    MIN = 10  # 变量的最小值和最大值
    MAX = 50
    uav_vel = MIN
    up_range = 10

    # 判断BS位置类型和pad_num来设置BS坐标和pads坐标
    BS, pads = get_position(bs_type, pad_num, area_size)

    # ---对不同的速度进行实验---
    print("-----Variable：Velocity-----")
    ARCS_res = []
    PWCS_res = []
    while uav_vel <= MAX:
        # ---得到变量文件下的文件，进行一组10次平均实验---
        print("the velocity of UAV is " + str(uav_vel))
        ARCS_res, PWCS_res = average_experiment(file_names, BS, pads, uav_max_E, uav_vel, uav_vel, ARCS_res, PWCS_res)
        uav_vel = uav_vel + up_range
    # 打印结果
    print('ARCS_res', ARCS_res)
    print('PWCS_res', PWCS_res)

    # ---输出文字日志---
    save_file = '-' + 'velocity' + '-' + str(node_type) + '-' + str(bs_type) + '.txt'
    top_write = "Variable index: The velocity of UAV(" + str(MIN) + "-" + str(MAX) + ")" + "\n"
    parameters = "The type of SNs: " + str(node_type) + "\n" \
                 + "The position of BS: " + str(bs_type) + "\n" \
                 + "The number of SNs: " + str(number) + "\n" \
                 + "The number of pads: " + str(3) + "\n" \
                 + "The Region size: " + str(area_size) + "\n" \
                 + "The max energy of UAV: " + str(uav_max_E) + "\n" \
                 + "The velocity of UAV: " + "[change]" + "\n"
    output_log(log_dir, save_file, top_write, parameters, [ARCS_res, PWCS_res], ['ARCS_res', 'PWCS_res'])

    return ARCS_res, PWCS_res

def experiment_entry(experiment_type, data_file_dir, result_dir, pad_num, uav_vel, uav_max_E):
    """
    实验入口，针对4种不同网络环境开始实验
    """
    node_types = ['NORMAL', 'NORMAL', 'GMM', 'GMM']
    bs_types = ['CENTER', 'ISOLATED_IN', 'CENTER', 'ISOLATED_IN']

    if experiment_type == 'number':
        number_results_ARCS = []
        number_results_PWCS = []
        for i in range(len(node_types)):
            ARCS_res, PWCS_res = experiment_number(data_file_dir, node_types[i], bs_types[i], pad_num, uav_vel, uav_max_E, result_dir['log'])
            number_results_ARCS.append(ARCS_res)
            number_results_PWCS.append(PWCS_res)
        xlsx_dir = result_dir['datas'] + '/number.xlsx'
        output_xlsx(number_results_ARCS, number_results_PWCS, xlsx_dir)

    if experiment_type == 'size':
        size_results_ARCS = []
        size_results_PWCS = []
        for i in range(len(node_types)):
            ARCS_res, PWCS_res = experiment_size(data_file_dir, node_types[i], bs_types[i], pad_num, uav_vel, uav_max_E, result_dir['log'])
            size_results_ARCS.append(ARCS_res)
            size_results_PWCS.append(PWCS_res)
        xlsx_dir = result_dir['datas'] + '/size.xlsx'
        output_xlsx(size_results_ARCS, size_results_PWCS, xlsx_dir)

    if experiment_type == 'pad':
        pad_results_ARCS = []
        pad_results_PWCS = []
        for i in range(len(node_types)):
            ARCS_res, PWCS_res = experiment_pad(data_file_dir, node_types[i], bs_types[i], pad_num, uav_vel, uav_max_E, result_dir['log'])
            pad_results_ARCS.append(ARCS_res)
            pad_results_PWCS.append(PWCS_res)
        xlsx_dir = result_dir['datas'] + '/pad.xlsx'
        output_xlsx(pad_results_ARCS, pad_results_PWCS, xlsx_dir)

    if experiment_type == 'energy':
        energy_results_ARCS = []
        energy_results_PWCS = []
        for i in range(len(node_types)):
            ARCS_res, PWCS_res = experiment_energy(data_file_dir, node_types[i], bs_types[i], pad_num, uav_vel, uav_max_E, result_dir['log'])
            energy_results_ARCS.append(ARCS_res)
            energy_results_PWCS.append(PWCS_res)
        xlsx_dir = result_dir['datas'] + '/energy.xlsx'
        output_xlsx(energy_results_ARCS, energy_results_PWCS, xlsx_dir)

    if experiment_type == 'velocity':
        velocity_results_ARCS = []
        velocity_results_PWCS = []
        for i in range(len(node_types)):
            ARCS_res, PWCS_res = experiment_velocity(data_file_dir, node_types[i], bs_types[i], pad_num, uav_vel, uav_max_E, result_dir['log'])
            velocity_results_ARCS.append(ARCS_res)
            velocity_results_PWCS.append(PWCS_res)
        xlsx_dir = result_dir['datas'] + '/velocity.xlsx'
        output_xlsx(velocity_results_ARCS, velocity_results_PWCS, xlsx_dir)

def main():
    # 实验数据地址
    data_dir = {'number': '../data/number/',
                'size': '../data/size/',
                'pad': '../data/pad/',
                'energy': '../data/energy/',
                'velocity': '../data/velocity/'}

    # 实验结果地址
    result_dir = {'log': '../result/logs/', 'datas': '../result/datas/', 'fig': '../result/figures/'}

    # 默认参数设置
    pad_num = 3  # 默认pad数量为3
    uav_vel = 20  # 默认uav速度为20m/s
    uav_max_E = 60000  # 默认UAV能量默认为60000

    # ---针对不同变量开始实验---
    experiment_entry('number', data_dir['number'], result_dir, pad_num, uav_vel, uav_max_E)
    # experiment_entry('size', data_dir['size'], result_dir, pad_num, uav_vel, uav_max_E)
    # experiment_entry('pad', data_dir['pad'], result_dir, pad_num, uav_vel, uav_max_E)
    # experiment_entry('energy', data_dir['energy'], result_dir, pad_num, uav_vel, uav_max_E)
    # experiment_entry('velocity', data_dir['velocity'], result_dir, pad_num, uav_vel, uav_max_E)

if __name__ == '__main__':
    main()

