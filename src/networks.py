########################################################################
# Author(s):    Zhichao Yang
# Date:         
# Desc:         Network models for GNSS-based position corrections
########################################################################
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm

import gnss_lib.coordinates as coord

########################################################
# PCGCNN
class PCGCNN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels,
                 num_layers=2,
                 detach_prev_state=True,  # 控制 Kalman-like vs RNN-like
                 similarity_threshold=0.9):
        super(PCGCNN, self).__init__()

        self.detach_prev_state = detach_prev_state
        self.similarity_threshold = similarity_threshold
        
        # 输入映射层
        self.input_linear = nn.Linear(in_channels, hidden_channels)

        # 构造 SAGEConv 层
        self.gnn_layers = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])

        # 批归一化
        self.bn = BatchNorm(hidden_channels)

        # 输出层
        self.output_linear = nn.Linear(hidden_channels, out_channels)

    def build_graph(self, h_prev, x_now, meta):
        """
        构造图神经网络的输入图结构。
    
        参数：
        - h_prev: [3,] 上一时刻的预测位置（非节点状态），若为 None 则不构造预测位置
        - x_now: [N, D] 当前观测特征（不包含 PPR）
        - meta: 包含 sat_pos, delta_position, exp_pseudorange
    
        返回：
        - x_with_ppr: [N, D+1]，每个节点的新特征（前面追加 PPR）
        - edge_index: [2, E]，图连接（全连接）
        """
        sat_pos = meta['sat_pos']        # [N, 3]
        exp_pseudo = meta['exp_pseudorange']  # [N]
        N = x_now.size(0)
        guess_prev=meta['guess_prev']

        if h_prev is not None:
            if self.detach_prev_state:
                h_prev = h_prev.detach()
            # 预测位置 = 上一时刻预测 + 本时刻 delta
            ref_local = coord.LocalCoord.from_ecef(guess_prev[:3])
            ned_prev=ref_local.ecef2ned(guess_prev[:3, None])[:, 0]
            pred_position = ref_local.ned2ecef(ned_prev+h_prev) + meta['delta_position']  # [3]
            # 广播计算预测位置到每个卫星的距离
            dists = torch.norm(sat_pos - pred_position.unsqueeze(0), dim=1)  # [N]
            ppr = dists - exp_pseudo  # [N]
        else:
        # 没有预测位置，用原始特征里的伪距残差（第一列）作为 PPR
            ppr = x_now[:, 0]  # [N]

    # 拼接 PPR 到特征前
        x_with_ppr = torch.cat([ppr.unsqueeze(1), x_now], dim=1)  # [N, D+1]

        sat_type=meta['sat_type']
        
        N = x_with_ppr.size(0)
        device = x_with_ppr.device
        row_list = []
        col_list = []

        # 你可以挑选部分维度作为计算余弦相似度的 basis
        # 这里我们假设 [0:4] 是你想用于相似度计算的子向量（可以改）
        feat_for_sim = x_with_ppr[:, :4]  # [N, 4]
        feat_norm = F.normalize(feat_for_sim, dim=1)  # L2 normalize

        cos_sim = torch.matmul(feat_norm, feat_norm.t())  # [N, N]
        sim_thresh = self.similarity_threshold  # 例如：0.9

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                same_constellation = sat_type[i] == sat_type[j]
                high_similarity = cos_sim[i, j] > sim_thresh
                if same_constellation or high_similarity:
                    row_list.append(i)
                    col_list.append(j)

        edge_index = torch.tensor([row_list, col_list], dtype=torch.long, device=device)

        return x_with_ppr, edge_index

    def forward(self, h_prev, x_now, meta):
        """
        h_prev: [N, H] 或 None
        x_now: [N, D]
        return: h_now: [N, H], output: [N, 3]
        """

        # 构图 & 融合
        h_combined, edge_index = self.build_graph(h_prev, x_now, meta)

        # 逐层 GNN
        h = h_combined
        for conv in self.gnn_layers:
            h = F.relu(conv(h, edge_index))

        h = self.bn(h)

        # 输出结果
        out = self.output_linear(h)
        return h, out


########################################################
# Set Transformer (modified implementation)

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = nn.MultiheadAttention(dim_out, num_heads)
        self.fc_q = nn.Linear(dim_in, dim_out)
        self.fc_k = nn.Linear(dim_in, dim_out)
        self.fc_v = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        Q = self.fc_q(X)
        K, V = self.fc_k(X), self.fc_v(X)
        out, wts = self.mab(Q, K, V)
        return out

# class ISAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
#         super(ISAB, self).__init__()
#         self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
#         nn.init.xavier_uniform_(self.I)
#         self.mab0 = nn.MultiheadAttention(dim_out, num_heads, kdim=dim_in, vdim=dim_out)
#         self.mab1 = nn.MultiheadAttention(dim_in, num_heads, kdim=dim_out, vdim=dim_out)

#     def forward(self, X):
#         H, _ = self.mab0(self.I.repeat(X.size(0), 1, 1), X, X)
#         out, _ = self.mab1(X, H, H)
#         return out

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(num_seeds, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = nn.MultiheadAttention(dim, num_heads)

    def forward(self, X, src_key_padding_mask=None):
        Q = self.S.repeat(1, X.size(1), 1)
        out, _ = self.mab(Q, X, X, key_padding_mask=src_key_padding_mask)
        return out

class Net_Snapshot(torch.nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=64, num_heads=4):
        super(Net_Snapshot, self).__init__()
#         self.enc = nn.Sequential(
#                 SAB(dim_input, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads))
        encoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2*dim_hidden, dropout=0.0)
        decoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2*dim_hidden, dropout=0.0)
        self.feat_in = nn.Sequential(
                        nn.Linear(dim_input, dim_hidden),
                    )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.dec = nn.Sequential(
#                 PMA(dim_hidden, num_heads, num_outputs),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 nn.Linear(dim_hidden, dim_output))
        self.pool = PMA(dim_hidden, num_heads, num_outputs)
        self.dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.feat_out = nn.Sequential(
                    nn.Linear(dim_hidden, dim_output)
                    )

    def forward(self, x, pad_mask=None):
        x = self.feat_in(x)
        x = self.enc(x, src_key_padding_mask=pad_mask)
        x = self.pool(x, src_key_padding_mask=pad_mask)
        x = self.dec(x)
        out = self.feat_out(x)
        return torch.squeeze(out, dim=0)


########################################################
# DeepSets (src: https://github.com/yassersouri/pytorch-deep-sets)       
class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)
        # sum up the representations
        x = torch.sum(x, dim=0, keepdim=False)
        # compute the output
        out = self.rho.forward(x)
        return out

class SmallPhi(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SmallRho(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class DeepSetModel(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        phi = SmallPhi(self.input_size, self.hidden_size)
        rho = SmallPhi(self.hidden_size, self.output_size)
        self.net = InvariantModel(phi, rho)
    
    def forward(self, x, pad_mask=None):
        out = self.net.forward(x)
        return out