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

def solve_vel(pseudorange_rate, los, vX, vY, vZ, e=1e-3):
    """
    解算接收机速度和钟漂速率，返回 [v_x, v_y, v_z, cdt_dot].
    Inputs:
      - pseudorange_rate: array-like, shape (N,), 伪距率观测 (m/s)
      - los: array-like, shape (N,3), LOS 单位向量
      - vX, vY, vZ: array-like, shape (N,), 卫星速度 (m/s)
      - e: 收敛阈值
    """
    # 转成 numpy
    pr = np.asarray(pseudorange_rate)
    los = np.asarray(los)
    vX = np.asarray(vX)
    vY = np.asarray(vY)
    vZ = np.asarray(vZ)

    if pr.size < 4:
        return np.empty(4)

    # 初始猜测：接收机零速，零钟漂
    vx, vy, vz, cdt_dot = 0., 0., 0., 0.

    def f(vars):
        vx, vy, vz, cdt_dot = vars
        # 预测伪距率
        pred = (los[:,0] * (vX - vx) +
                los[:,1] * (vY - vy) +
                los[:,2] * (vZ - vz) +
                cdt_dot)
        return pr - pred  # 残差

    def df(vars):
        # 对 [vx,vy,vz,cdt_dot] 的雅可比
        n = pr.size
        J = np.zeros((n,4))
        # ∂f/∂vx = -∂pred/∂vx = -(-los[:,0]) = +los[:,0]
        J[:,0] =  los[:,0]
        J[:,1] =  los[:,1]
        J[:,2] =  los[:,2]
        # ∂pred/∂cdt_dot = 1  =>  ∂f/∂cdt_dot = -1
        J[:,3] = -1.0
        return J

    x0 = np.array([vx, vy, vz, cdt_dot])
    x_fix, res_err = newton_raphson(f, df, x0, e=e)
    return x_fix