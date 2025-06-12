########################################################################
# Author(s):    Zhichao Yang
# Date:         
# Desc:         Network models for GNSS-based position corrections
########################################################################
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm, global_mean_pool

import gnss_lib.coordinates_gpu as coord

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
        #self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.input_gnn = SAGEConv(in_channels, hidden_channels) 

        # 构造 SAGEConv 层
        self.gnn_layers = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])

        # 批归一化
        self.bn = BatchNorm(hidden_channels)

        # 全局池化层（将节点特征聚合成图特征）
        self.pool = global_mean_pool  # 或其他池化函数如 global_max_pool

        # 输出层
        self.output_linear = nn.Linear(hidden_channels, out_channels)

    def build_graph(self, h_prev, x_now, meta, pad_mask=None):
        """
        重构后的构图函数（集中处理所有数据逻辑）
        
        参数：
        - h_prev: [B,3] 或 None
        - x_now: [L_max, B, D] 原始输入特征
        - meta: 包含批处理后的动态/静态字段
        - pad_mask: [B, L_max] 填充掩码
        
        返回：
        - x_with_ppr: [num_valid_nodes, D+1]
        - edge_index: [2, E]
        - batch: [num_valid_nodes] 图划分标识
        """
        # 1. 处理原始输入维度
        L_max, B, D = x_now.shape
        device = x_now.device
        
        # 2. 筛选有效节点
        valid_mask = ~pad_mask.reshape(-1)  # [B*L_max]
        valid_indices = torch.where(valid_mask)[0]
        num_valid_nodes = len(valid_indices)
        
        # 3. 处理特征（含PPR计算）
        x_flat = x_now.permute(1,0,2).reshape(B*L_max, D)  # [B*L_max, D]
        valid_x = x_flat[valid_mask]  # [num_valid_nodes, D]
        
        if h_prev is not None:
            # PPR计算（批量处理）
            pred_positions = []
            for b in range(B):
                ref_local = coord.LocalCoordGPU.from_ecef(meta['guess_prev'][b,:3])
                ned_prev = ref_local.ecef2ned(meta['guess_prev'][b,:3])
                pred_pos = ref_local.ned2ecef(ned_prev + h_prev[b]) + meta['delta_position'][b,:3]
                pred_positions.append(pred_pos)
            pred_position = torch.stack(pred_positions)  # [B,3]
            
            # 计算距离（仅对有效节点）
            sat_pos_valid = meta['sat_pos'].reshape(B*L_max, 3)[valid_mask]  # [num_valid,3]
            pred_expanded = pred_position.repeat_interleave(L_max, dim=0)[valid_mask]  # [num_valid,3]
            dists = torch.norm(sat_pos_valid - pred_expanded, dim=1) + meta['guess'][:,-1].repeat_interleave(L_max)[valid_mask]
            ppr = meta['PrM'].reshape(B*L_max)[valid_mask] - dists  # [num_valid]
        else:
            ppr = valid_x[:, 0]  # 使用第一列作为PPR
        
        # 4. 构建最终特征
        x_with_ppr = torch.cat([ppr.unsqueeze(1), valid_x], dim=1)  # [num_valid, D+1]
        
        # 5. 构图（多图处理）
        batch = torch.repeat_interleave(
            torch.arange(B, device=device),
            (~pad_mask).sum(dim=1)  # 精确计算每个样本的有效节点数
        )
        
        edge_indices = []
        sat_type_valid = meta['sat_type'].reshape(B*L_max)[valid_mask]  # [num_valid]
        
        for b in range(B):
            sample_mask = (batch == b)
            current_nodes = torch.where(sample_mask)[0]
            if len(current_nodes) == 0:
                continue
                
            # 当前样本的特征和类型
            feat = x_with_ppr[sample_mask]
            types = sat_type_valid[sample_mask]
            
            # 计算相似度
            feat_norm = F.normalize(feat[:,:4], dim=1)
            cos_sim = torch.mm(feat_norm, feat_norm.t())
            
            # 构建边
            type_mask = (types.unsqueeze(1) == types.unsqueeze(0)).to(device)
            connect_mask = (cos_sim > self.similarity_threshold) | type_mask
            rows, cols = torch.where(torch.triu(connect_mask, diagonal=1))
            
            # 转换为全局索引并添加双向边
            global_idx = current_nodes[torch.stack([rows, cols])]
            edge_indices.append(global_idx)
            edge_indices.append(global_idx.flip(0))
        
        edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty(2,0, device=device)
        
        return x_with_ppr, edge_index, batch

    def forward(self, h_prev, x_now, meta, pad_mask=None):
        """
        简化后的前向传播
        """
        # 所有数据处理移至build_graph
        x_with_ppr, edge_index, batch = self.build_graph(
            h_prev, 
            x_now, 
            meta,
            pad_mask
        )
        
        # GNN处理
        h = x_with_ppr
        h = F.relu(self.input_gnn(h, edge_index))
        h = self.bn(h)
        
        for conv in self.gnn_layers:
            h = F.relu(conv(h, edge_index))
            h = self.bn(h)
        
        # 池化（使用build_graph返回的精确batch划分）
        h_pooled = self.pool(h, batch)
        return self.output_linear(h_pooled)


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