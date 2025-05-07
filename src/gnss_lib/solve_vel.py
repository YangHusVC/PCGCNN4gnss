########################################################################
# Author(s):    Zhichao Yang
# Date:         5 March, 2025
# Desc:         Code to find Newton Raphson PNT solution
########################################################################
import numpy as np
from . import constants

def newton_raphson(f, df, x0, e=1e-3, lam=1.):
    delta_x = np.ones_like(x0)
    while np.sum(np.abs(delta_x))>e:
        delta_x = lam*(np.linalg.pinv(df(x0)) @ f(x0))
        x0 = x0 - delta_x
    return x0, np.linalg.norm(f(x0))

def get_glonass_frequency(svid):
    # 伪码 svid 转换为通道号 k
    k = svid - 64 - 8  # 对应 GLO_G1：svid = 72 --> k=0
    return 1602e6 + k * 0.5625e6

signal_frequency_map = {
    'GPS_L1': 1.57542e9,
    'GPS_L5': 1.17645e9,
    'GAL_E1': 1.57542e9,
    'GAL_E5A': 1.17645e9,
}

def solve_vel(cfreq, signalType,svid,los,vX,vY,vZ,e=1e-3):
    #排除了GLOSNASS信号，因为数据集不能推测其中心频率
    #mask = np.isin(signalType, ['GPS_L1', 'GAL_E1'])
    mask = np.array(signalType) != 'GLO_G1'
    cfreq = cfreq[mask]
    signalType = np.array(signalType)[mask]
    svid = np.array(svid)[mask]
    los = los[mask]
    vX = np.array(vX)[mask]
    vY = np.array(vY)[mask]
    vZ = np.array(vZ)[mask]

    lambdas = []
    f_carriers=[]
    for sig, sv in zip(signalType, svid):
        if sig == 'GLO_G1':
            f_carrier = get_glonass_frequency(sv)
        else:
            f_carrier = signal_frequency_map.get(sig, np.nan)
        if np.isnan(f_carrier):
            raise ValueError(f"Unknown signal type: {sig}")
        lambdas.append(constants.LIGHTSPEED / f_carrier)
        f_carriers.append(f_carrier)
    lambdas = np.array(lambdas)
    f_carriers=np.array(f_carriers)
    cfreq=np.array(cfreq)
    doppler_shift=cfreq-f_carriers

    # 卫星速度向量 shape: (n, 3)
    v_sat = np.stack([vX, vY, vZ], axis=1)

    # 初始估计：接收机速度 vx, vy, vz 和钟频偏 cdt
    vx, vy, vz, cdt = 0., 0., 0., 0.
    x0 = np.array([vx, vy, vz, cdt])

    def f(vars):
        vx, vy, vz, cdt = vars
        v_rx = np.array([vx, vy, vz])
        relative_vel = v_sat - v_rx  # 卫星速度 - 接收机速度
        doppler_pred = -np.sum(relative_vel * los, axis=1) / lambdas + cdt
        residual = doppler_shift - doppler_pred
        return residual

    def df(vars):
        J = np.zeros((len(cfreq), 4))
        for i in range(len(cfreq)):
            # 速度分量的导数应为负号
            J[i, 0] = -los[i, 0] / lambdas[i]
            J[i, 1] = -los[i, 1] / lambdas[i]
            J[i, 2] = -los[i, 2] / lambdas[i]
            J[i, 3] = -1
        return J  # 不再返回负矩阵！
        #return -J  # 注意是负的导数方向
    v_fix, res_err = newton_raphson(f, df, x0, e=e)
    return v_fix  # [vx, vy, vz, cdt]