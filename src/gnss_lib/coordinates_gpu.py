import torch
import math

# WGS84 椭球参数 (直接定义为 GPU 张量)
a = torch.tensor(6378137.0, dtype=torch.float32).cuda()          # 长半轴 (m)
b = torch.tensor(6356752.314245, dtype=torch.float32).cuda()     # 短半轴 (m)
esq = torch.tensor(6.69437999014 * 0.001, dtype=torch.float32).cuda()                                         # 第一偏心率的平方 (GPU 张量)
e1sq = torch.tensor(6.73949674228 * 0.001, dtype=torch.float32).cuda()                                     # 第二偏心率的平方 (GPU 张量)

def ecef2geodetic_gpu(ecef: torch.Tensor, radians: bool = False) -> torch.Tensor:
    input_shape = ecef.shape
    if ecef.dim() == 1:  # 直接处理 1D 输入
        ecef = ecef.unsqueeze(0)  # 形状 [1, 3]
    else:
        ecef = ecef.reshape(-1, 3)
    
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    ratio = 1.0 if radians else (180.0 / math.pi)

    # Ferrari's 算法 (矢量化实现)
    r = torch.sqrt(x**2 + y**2)
    F = 54 * (b**2) * z**2
    G = r**2 + (1 - esq) * z**2 - esq * (a**2 - b**2)
    
    C = (esq**2 * F * r**2) / (G**3 + 1e-12)  # 避免除零
    S = torch.pow(C + torch.sqrt(C**2 + 2*C + 1e-12), 1/3)  # 替代 np.cbrt
    
    P = F / (3 * (S + 1/(S + 1e-12))**2 * (G**2 + 1e-12))
    Q = torch.sqrt(1 + 2 * esq**2 * P)
    
    r_0 = -P * esq * r / (Q + 1) + torch.sqrt(
        0.5 * a**2 * (1 + 1/Q) - 
        P * (1 - esq) * z**2 / (Q * (Q + 1)) - 
        0.5 * P * r**2
    )
    
    U = torch.sqrt((r - esq * r_0)**2 + z**2)
    V = torch.sqrt((r - esq * r_0)**2 + (1 - esq) * z**2)
    
    Z_0 = (b**2 * z) / (a * V + 1e-12)
    h = U * (1 - b**2 / (a * V + 1e-12))
    
    lat = ratio * torch.atan((z + e1sq * Z_0) / (r + 1e-12))  # 避免除零
    lon = ratio * torch.atan2(y, x)
    
    # 重组输出张量
    geodetic = torch.stack([lat, lon, h], dim=-1)
    return geodetic.view(*input_shape)


class LocalCoordGPU:
    """
    GPU 版本的局部坐标系转换工具 (NED 坐标系)
    所有输入和中间计算均在 GPU 张量上完成
    """
    def __init__(self, init_geodetic: torch.Tensor, init_ecef: torch.Tensor):
        """
        init_geodetic: [lat(deg), lon(deg), alt(m)] 形状为 [3] 的 GPU 张量
        init_ecef: [x, y, z] ECEF 坐标，形状为 [3] 的 GPU 张量
        """
        self.init_ecef = init_ecef
        lat = torch.deg2rad(init_geodetic[0])
        lon = torch.deg2rad(init_geodetic[1])
        
        # 构建 NED 转换矩阵 (GPU 张量)
        self.ned2ecef_matrix = torch.tensor([
            [-torch.sin(lat)*torch.cos(lon), -torch.sin(lon), -torch.cos(lat)*torch.cos(lon)],
            [-torch.sin(lat)*torch.sin(lon),  torch.cos(lon), -torch.cos(lat)*torch.sin(lon)],
            [ torch.cos(lat),                 0,              -torch.sin(lat)]
        ], dtype=torch.float64, device=init_ecef.device)
        
        self.ecef2ned_matrix = self.ned2ecef_matrix.T

    @classmethod
    def from_ecef(cls, init_ecef: torch.Tensor):
        """直接从 ECEF 坐标初始化"""
        init_geodetic = ecef2geodetic_gpu(init_ecef)
        return cls(init_geodetic, init_ecef)

    def ecef2ned(self, ecef: torch.Tensor) -> torch.Tensor:
        """将 ECEF 坐标转换为 NED 坐标（仅支持单个3D向量输入）"""
        assert ecef.shape == torch.Size([3]), "输入必须是形状为 [3] 的一维张量"
    
        # 计算差值并转换为列向量 [3,1]
        diff = (ecef - self.init_ecef).unsqueeze(-1)  # 形状 [3,1]
    
        # 矩阵乘法: [3,3] * [3,1] -> [3,1]
        ned = torch.mm(self.ecef2ned_matrix, diff)  # 形状 [3,1]
        return ned.squeeze(-1)  # 压缩回 [3]

    def ned2ecef(self, ned: torch.Tensor) -> torch.Tensor:
        """将 NED 坐标转换回 ECEF 坐标（仅支持单个3D向量输入）"""
        assert ned.shape == torch.Size([3]), "输入必须是形状为 [3] 的一维张量"
    
        # 转换为列向量 [3,1]
        ned_col = ned.unsqueeze(-1)  # 形状 [3,1]
    
        # 矩阵乘法: [3,3] * [3,1] -> [3,1]
        rotated = torch.mm(self.ned2ecef_matrix, ned_col)  # 形状 [3,1]
        ecef = rotated.squeeze(-1) + self.init_ecef  # 形状 [3]
        return ecef