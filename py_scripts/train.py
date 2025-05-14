########################################################################
# Author(s):    Zhichao Yang
# Date:         
# Desc:                 
########################################################################
import sys, os, csv, datetime
parent_directory = os.path.split(os.getcwd())[0]
parent_directory = os.path.join(parent_directory, 'PCGCNN4gnss')
src_directory = os.path.join(parent_directory, 'src')
data_directory = os.path.join(parent_directory, 'datasets')
sys.path.insert(0, src_directory)
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np # linear algebra
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from android_dataset import Android_GNSS_Dataset
from networks import PCGCNN, Net_Snapshot, DeepSetModel


#用于数据增强
class RandomTailDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, start_limit_ratio=0.2):
        """
        base_dataset: 一个 Subset 或 Dataset，假设是训练集部分（已有序）
        start_limit_ratio: 起始点可选范围（比例），如 0.2 表示从前 20% 内取起点
        """
        self.base_dataset = base_dataset
        self.total_len = len(base_dataset)
        self.start_limit = int(start_limit_ratio * self.total_len)
        self.start_idx = np.random.randint(0, max(self.start_limit, 1))  # 防止为0

        self.indices = list(range(self.start_idx, self.total_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

def collate_feat(batch):
    #不用于GNN
    sorted_batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)
    features = [x['features'] for x in sorted_batch]
    features_padded = torch.nn.utils.rnn.pad_sequence(features)
    L, N, dim = features_padded.size()
    pad_mask = np.zeros((N, L))
    for i, x in enumerate(features):
        pad_mask[i, len(x):] = 1
    pad_mask = torch.Tensor(pad_mask).bool()
    correction = torch.Tensor([x['true_correction'] for x in sorted_batch])
    guess = torch.Tensor([x['guess'] for x in sorted_batch])
    retval = {
            'features': features_padded,
            'true_correction': correction,
            'guess': guess
        }
    return retval, pad_mask  

def collate_feat0(batch):
    # 按 features 序列长度从大到小排序
    sorted_batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)

    # 拼接特征序列：无 padding，直接 stack（假设每个 sample 长度一致）
    features = [x['features'] for x in sorted_batch]
    features_tensor = torch.stack(features, dim=1)  # shape: [L, N, dim] 假设序列等长

    correction = torch.Tensor([x['true_correction'] for x in sorted_batch])
    guess = torch.stack([x['guess'] for x in sorted_batch])

    # === 处理 meta 字段 === #
    delta_position = torch.stack([x['delta_position'] for x in sorted_batch])
    sat_pos = torch.stack([x['sat_pos'] for x in sorted_batch])
    exp_pseudorange = torch.stack([x['exp_pseudorange'] for x in sorted_batch])
    sat_type = torch.stack([x['sat_type'] for x in sorted_batch])
    PrM = torch.stack([x['PrM'] for x in sorted_batch])

    retval = {
        'features': features_tensor,
        'true_correction': correction,
        'guess': guess,
        'delta_position': delta_position,
        'sat_pos': sat_pos,
        'exp_pseudorange': exp_pseudorange,
        'sat_type': sat_type,
        'PrM':  PrM
    }

    pad_mask = None  # 不再使用 pad，所以设为 None（或者 torch.zeros([N, L], dtype=torch.bool)）

    return retval, pad_mask

def test_eval(val_loader, net, loss_func):
    """
    验证阶段评估函数，保持轨迹连续性，支持 h_prev 和 guess_prev 的传播。
    """
    stats_val = []
    total_loss = 0
    net.eval()

    h_prev = None
    guess_prev = torch.zeros(3) #WWWWWWWWWWWWWWWWWWWW
    delta_position_prev=torch.zeros(3)

    with torch.no_grad():
        for sample_batched, pad_mask in tqdm(val_loader, desc='test', leave=False):
            _sample_batched = sample_batched
            if pad_mask!=None:
                pad_mask = pad_mask.cuda()

            x = _sample_batched['features'].float().cuda()
            y = _sample_batched['true_correction'].float().cuda()

            # 初始化 guess_prev（第一次用 dataset 提供的 guess）
            if guess_prev is None:
                guess_prev = _sample_batched['guess'].float().cuda()

            meta={
                'delta_position': delta_position_prev.float().cuda(),
                'sat_pos':_sample_batched['sat_pos'].float().cuda(),
                #'exp_pseudorange':_sample_batched['exp_pseudorange'].float().cuda(),
                'PrM':_sample_batched['PrM'].float().cuda(),
                'sat_type':_sample_batched['sat_type'].long(),
                'guess_prev':guess_prev.float().cuda(),#torch.tensor(guess_prev).clone().detach().float().cuda() #之前是wls算出来的，不需要求梯度
                'guess':_sample_batched['guess'][0].float().cuda()
            }

            # 前向推理
            pred_correction= net(x_now=x.squeeze(1), h_prev=h_prev,meta=meta)
            h_prev = pred_correction 

            # 更新 guess_prev 为当前的 guess（你在 dataloader 提供）
            guess_prev=_sample_batched['guess']
            delta_position_prev=_sample_batched['delta_position']

            loss = loss_func(pred_correction, y)
            total_loss += loss.item()

            stats_val.append((y - pred_correction).cpu().numpy())

    avg_abs_error = np.mean(np.abs(np.array(stats_val)), axis=0)#np.mean(np.abs(np.concatenate(stats_val, axis=0)), axis=0)
    avg_loss = total_loss / len(stats_val)

    return avg_abs_error, avg_loss


@hydra.main(config_path="../config", config_name="train_gsdc_2021")
def main(config: DictConfig) -> None:
    data_config = {
    "root": data_directory,
    "raw_data_dir" : config.raw_data_dir,
    "data_dir": config.data_dir,
    # "initialization_dir" : "initialization_data",
    # "info_path": "data_info.csv",
    "max_open_files": config.max_open_files,
    "guess_range": [config.pos_range_xy, config.pos_range_xy, config.pos_range_z, config.clk_range, config.vel_range_xy, config.vel_range_xy, config.vel_range_z, config.clkd_range],
    "history": config.history,
    "seed": config.seed,
    "chunk_size": config.chunk_size,
    "max_sats": config.max_sats,
    "bias_fname": config.bias_fname,
    }
    
    print('Initializing dataset')
    
    dataset = Android_GNSS_Dataset(data_config)
    print('processsed data saved')
    
    total_len = len(dataset)
    train_len = int(config.frac * total_len)  # config.frac = 0.8 之类

    train_set = torch.utils.data.Subset(dataset, range(train_len))
    val_set = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    #无数据增强，debug用
    dataloader = DataLoader(train_set, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers,collate_fn=collate_feat0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, num_workers=0,collate_fn=collate_feat0)
    print('Initializing network: ', config.model_name)
    if config.model_name == "PCGCNN":
        net = PCGCNN(in_channels=6, hidden_channels=64, out_channels=3, detach_prev_state=config.detach_prev_state,similarity_threshold=config.similarity_threshold)
    elif config.model_name=='Set Transformer':
        net = Net_Snapshot(train_set[0]['features'].size()[1], 1, len(train_set[0]['true_correction']))     # define the network
    else:
        raise ValueError('This model is not supported yet!')
    
    if not config.resume==0:
        net.load_state_dict(torch.load(os.path.join(data_directory, 'weights', config.resume)))
        print("Resumed: ", config.resume)
    
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), config.learning_rate)
    loss_func = torch.nn.MSELoss()
    count = 0
    fname = "android_" + config.prefix + "_"+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if config.writer:
        writer = SummaryWriter(os.path.join(data_directory, 'runs', fname))


    min_acc = 100
    
    for epoch in range(config.N_train_epochs):
        # TRAIN Phase

        '''
        #带数据增强
        train_rs_dataset = RandomTailDataset(train_set, start_limit_ratio=config.trainset_split_ratio) 
        dataloader = DataLoader(train_rs_dataset, batch_size=1,
                            shuffle=False, num_workers=config.num_workers)

        val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=0)'''
        net.train()
        h_prev = None
        guess_prev=torch.zeros(3)
        delta_position_prev=torch.zeros(3)
        for i, sample_batched in enumerate(dataloader):
            _sample_batched, pad_mask = sample_batched
            
            x = _sample_batched['features'].float().cuda()
            y = _sample_batched['true_correction'].float().cuda()
            

            meta={
                'delta_position': delta_position_prev.float().cuda(),
                'sat_pos':_sample_batched['sat_pos'].float().cuda(),
                #'exp_pseudorange':_sample_batched['exp_pseudorange'].float().cuda(),
                'PrM':_sample_batched['PrM'].float().cuda(),
                'sat_type':_sample_batched['sat_type'].long(),
                'guess_prev':guess_prev.float().cuda(),#torch.tensor(guess_prev).clone().detach().float().cuda() #之前是wls算出来的，不需要求梯度
                'guess':_sample_batched['guess'][0].float().cuda()
            }
            #pad_mask = pad_mask.cuda()
            pred_correction= net(x_now=x.squeeze(1), h_prev=h_prev,meta=meta)
            h_prev = pred_correction 
            loss = loss_func(pred_correction, y)
            if config.writer:
                writer.add_scalar("Loss/train", loss.item(), count)
                
            guess_prev=_sample_batched['guess']
            delta_position_prev=_sample_batched['delta_position']
            count += 1    
            
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        # TEST Phase
        net.eval()
        mean_acc, test_loss = test_eval(val_loader, net, loss_func)
        if config.writer:
            writer.add_scalar("Loss/test", test_loss, epoch)
        for j in range(len(mean_acc[0])):
            if config.writer:
                writer.add_scalar("Metrics/Acc_"+str(j), mean_acc[0, j], epoch)
        if np.sum(mean_acc) < min_acc:
            min_acc = np.sum(mean_acc)
            torch.save(net.state_dict(), os.path.join(data_directory, 'weights', fname))
        print('Training done for ', epoch)

if __name__=="__main__":
    main()