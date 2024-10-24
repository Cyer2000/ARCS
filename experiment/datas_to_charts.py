import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def draw_chart(data, parameter, chart_dir):
    """
    画实验图
    """
    # 将八个数据进行合并
    # data = [data1, data2, data3, data4, data5, data6, data7, data8]

    # 设置图片相关
    plt.rc('font', family='Times New Roman', size=20)
    plt.figure(figsize=(35, 20), dpi=100)
    plt.subplots_adjust(hspace=0.4)

    # 画子图
    for i in range(len(data)):
        if i % 2 == 0:
            x = np.array(data[i])[:, 0]
            y1 = np.array(data[i])[:, 1]
            y2 = np.array(data[i + 1])[:, 1]

            plt.subplot(2, 2, int(i / 2) + 1)
            plt.plot(x, y2, label='PWCS', linewidth=2.5, marker='s', markersize=16, c='hotpink')
            plt.plot(x, y1, label='ARCS', linewidth=2.5, marker='o', markersize=16, c='dodgerblue')

            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='lower right', borderpad=0.5, fontsize=36, markerscale=0.7)
            plt.xlabel(parameter, fontsize=44, labelpad=15)
            plt.ylabel("charging delay (s)", fontsize=44, labelpad=20)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y', useOffset=False, useMathText=True)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.title("(" + chr(97 + int(i / 2)) + ")", y=-0.3, fontsize=44, fontweight='bold')

    # 如果有地址的话，保存图片
    if chart_dir is not None:
        plt.savefig(chart_dir)
        plt.close()
    else:
        plt.show()  # 展示图片

    return


def extract_data(file_dir):
    """
    从excel文件中提取数据
    """
    data = []

    if file_dir == '../result/datas/energy.xlsx':
        df = pd.read_excel(file_dir)
        for i in range(len(df.values)):
            tmp_data = []
            for j in range(len(df.values[i])):
                print(df.values[i][j])
                x = float(df.values[i][j][1:-1].split(',')[0])
                y = float(df.values[i][j][1:-1].split(',')[1])
                tmp_data.append(np.array([x, y]).tolist())
            data.append(tmp_data)
        return data

    df = pd.read_excel(file_dir)
    for i in range(len(df.values)):
        tmp_data = []
        for j in range(len(df.values[i])):
            print(df.values[i][j])
            x = int(df.values[i][j][1:-1].split(',')[0])
            y = float(df.values[i][j][1:-1].split(',')[1])
            tmp_data.append(np.array([x, y]).tolist())
        data.append(tmp_data)

    return data


def main():

    # 实验数据地址
    num_dir = '../result/datas/number.xlsx'
    size_dir = '../result/datas/size.xlsx'
    pad_dir = '../result/datas/pad.xlsx'
    energy_dir = '../result/datas/energy.xlsx'

    # 保存地址
    num_chart_dir = '../result/charts//number.jpg'
    size_chart_dir = '../result/charts//size.jpg'
    pad_chart_dir = '../result/charts//pad.jpg'
    energy_chart_dir = '../result/charts//energy.jpg'

    # 提取数据->定参数（x坐标）->画图表
    # num_data = extract_data(num_dir)
    # num_parameter = 'the number of sensor nodes'
    # draw_chart(num_data, num_parameter, num_chart_dir)

    # size_data = extract_data(size_dir)
    # size_parameter = 'the size of the monitoring area(m)'
    # draw_chart(size_data, size_parameter, size_chart_dir)

    # pad_data = extract_data(pad_dir)
    # pad_parameter = 'the number of pre-deployed PADs'
    # draw_chart(pad_data, pad_parameter, pad_chart_dir)

    energy_data = extract_data(energy_dir)
    energy_parameter = 'the energy capacity of the UAV(J)'
    draw_chart(energy_data, energy_parameter, energy_chart_dir)


if __name__ == '__main__':
    main()
