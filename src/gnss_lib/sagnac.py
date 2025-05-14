import numpy as np
from . import constants

def apply_earth_rotation_correction(df, pr_col='PrM', 
                                   x_col='xSatPosM', y_col='ySatPosM', z_col='zSatPosM'):
    """
    对DataFrame中所有卫星坐标进行地球自转校正预处理
    新增列: 'xSatCorrected', 'ySatCorrected', 'zSatCorrected'
    """
    omega_e = 7.292115e-5  # 地球自转角速度 (rad/s)
    
    # 计算信号传播时间 (单位：秒)
    delta_t = df[pr_col].values / constants.LIGHTSPEED  # 向量化计算
    
    # 计算旋转角
    theta = omega_e * delta_t  # 每个卫星对应一个角度
    
    # 向量化旋转矩阵计算
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 原始坐标矩阵 [N x 3]
    sat_pos = df[[x_col, y_col, z_col]].values
    
    # 应用旋转矩阵（广播计算）
    x_corr = sat_pos[:,0] * cos_theta + sat_pos[:,1] * sin_theta
    y_corr = -sat_pos[:,0] * sin_theta + sat_pos[:,1] * cos_theta
    z_corr = sat_pos[:,2]
    
    # 存入DataFrame
    df['xSatCorrected'] = x_corr
    df['ySatCorrected'] = y_corr
    df['zSatCorrected'] = z_corr
    
    return df