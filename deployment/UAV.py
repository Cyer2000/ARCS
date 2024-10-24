# 功率就是单位时间的能量流（消耗或者传输）
import math

# 无人机旋翼详细参数
W = 20  # 飞机重量(N) G=mg
ρ = 1.225  # 空气密度(kg/m^3)
R = 0.4  # 旋翼半径(m)
A = 0.503  # (m^2) A≜πR^2
Ω = 300  # 叶片角速度（弧度/秒）
U_tip = 120  # 旋翼叶尖速度(m/s) U_tip≜ΩR
s = 0.05  # 旋翼坚固度
d_0 = 0.6  # 机身阻力比
k = 0.1  # 诱导功率增量校正系数
v_0 = 4.03  # 悬停时的平均旋翼诱导速度
δ = 0.012  # 剖面阻力系数


class UAV(object):
    def __init__(self, vel, max_E, pos, node_stop=2, pad_stop=20):
        """
        :param vel: 无人机飞行速度
        :param max_E: 无人机存储最大能量
        :param T_E: 能量阈值
        :param curr_E: 当前剩余能量
        :param P_mov: 飞行时能量功率
        :param P_hov: 悬停时能量功率
        :param E_η: 传输能量损耗
        :param pos: 位置信息
        :param node_stop: 在node停留充电时间
        :param pad_stop: 在pad停留补充电能时间
        :param trip_time: 路程时间长度
        :param trip_time: 路程时间长度
        :param E_pro: 无人机推进耗能（移动+悬停）
        :param E_wpt: 无人机传输耗能（充电）
        """
        self.vel = vel
        self.max_E = max_E
        self.curr_E = max_E
        self.E_η = 0.9
        self.pos = pos
        self.node_stop = node_stop
        self.pad_stop = pad_stop
        self.P_mov = 0
        self.P_hov = 0
        self.trip_time = 0
        self.E_pro = [0, 0]
        self.E_wpt = 0

    def computePower(self):
        """
        根据无人机的速度vel计算飞行和悬停功率
        :return: 飞行功率P_mov和悬停功率P_hov
        """
        P_0 = (δ / 8) * ρ * s * A * math.pow(Ω, 3) * math.pow(R, 3)
        P_i = (1 + k) * (math.pow(W, 3 / 2) / math.sqrt(2 * ρ * A))
        self.P_hov = P_0 + P_i
        part1 = P_0 * (1 + (3 * math.pow(self.vel, 2)) / (math.pow(U_tip, 2)))
        part2 = P_i * math.sqrt(math.sqrt(1 + math.pow(self.vel, 4) / (4 * math.pow(v_0, 4))) - math.pow(self.vel, 2) / (2 * math.pow(v_0, 2)))
        part3 = 0.5 * d_0 * ρ * s * A * math.pow(self.vel, 3)
        self.P_mov = part1 + part2 + part3
        # print(self.P_mov)
        # print(self.P_hov)

    def maxRadius(self, node_E):
        """
        计算无人机从pad最远达到充电节点的距离（作为圆贪婪半径）
        无人机总能量 = 无人机往返飞行耗能+无人机充电悬停耗能+无人机充电传输耗能
        :param node_E: 节点最大携带能量
        :return:
        """
        mov_RE = self.max_E - node_E - self.node_stop * self.P_hov  # 可供飞行所剩能量
        radius = ((mov_RE / self.P_mov) * self.vel) / 2  # 可飞行时间*飞行速度=最远飞行充电半径

        return radius

    def maxDistance(self):
        """
        无人机最大飞行距离：所有能量用于飞行
        :return:
        """
        distance = (self.max_E / self.P_mov) * self.vel
        return distance


