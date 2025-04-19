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


def collate_feat(batch):
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
        }#WWWWWWWWWarning
    return retval, pad_mask       

def test_eval(val_loader, net, loss_func):
    """
    验证阶段评估函数，保持轨迹连续性，支持 h_prev 和 guess_prev 的传播。
    """
    stats_val = []
    total_loss = 0
    net.eval()

    h_prev = None
    guess_prev = None

    with torch.no_grad():
        for sample_batched, pad_mask in tqdm(val_loader, desc='test', leave=False):
            _sample_batched = sample_batched
            pad_mask = pad_mask.cuda()

            x = _sample_batched['features'].float().cuda()
            y = _sample_batched['true_correction'].float().cuda()

            # 初始化 guess_prev（第一次用 dataset 提供的 guess）
            if guess_prev is None:
                guess_prev = _sample_batched['guess'].float().cuda()

            meta = {
                'delta_position': _sample_batched['delta_position'].float().cuda(),
                'sat_pos': _sample_batched['sat_pos'].float().cuda(),
                'exp_pseudorange': _sample_batched['exp_pseudorange'].float().cuda(),
                'sat_type': _sample_batched['sat_type'].long().cuda(),
                'guess_prev': guess_prev
            }

            # 前向推理
            pred_correction, h_prev = net(x, h_prev=h_prev, pad_mask=pad_mask, meta=meta)

            # 更新 guess_prev 为当前的 guess（你在 dataloader 提供）
            guess_prev = _sample_batched['guess'].float().cuda()

            loss = loss_func(pred_correction, y)
            total_loss += loss.item()

            stats_val.append((y - pred_correction).cpu().numpy())

    avg_abs_error = np.mean(np.abs(np.concatenate(stats_val, axis=0)), axis=0)
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
    
    train_set, val_set = torch.utils.data.random_split(dataset, [int(config.frac*len(dataset)), len(dataset) - int(config.frac*len(dataset))])
    dataloader = DataLoader(train_set, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers, collate_fn=collate_feat)#WWWWWWWWWWWWWWWWWWWWWWarning
    val_loader = DataLoader(val_set, batch_size=1, 
                            shuffle=False, num_workers=0, collate_fn=collate_feat)
    print('Initializing network: ', config.model_name)
    if config.model_name == "PCGCNN":
        net = PCGCNN(in_channels=9, hidden_channels=128, out_channels=3, detach_prev_state=config.detach_prev_state,similarity_threshold=config.similarity_threshold)
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


    min_acc = 1000000
    
    for epoch in range(config.N_train_epochs):
        # TRAIN Phase
        net.train()
        h_prev = None
        guess_prev=None
        for i, sample_batched in enumerate(dataloader):
            _sample_batched, pad_mask = sample_batched
            
            x = _sample_batched['features'].float().cuda()
            y = _sample_batched['true_correction'].float().cuda()
            

            meta={
                'delta_position': _sample_batched['delta_position'].float().cuda(),
                'sat_pos':_sample_batched['sat_pos'].float().cuda(),
                'exp_pseudorange':_sample_batched['exp_pseudorange'].float().cuda(),
                'sat_type':_sample_batched['sat_type'].long(),
                'guess_prev':torch.Tensor(guess_prev).float.cuda
            }
            #pad_mask = pad_mask.cuda()
            pred_correction, h_prev = net(x, h_prev=h_prev,meta=meta)
            loss = loss_func(pred_correction, y)
            if config.writer:
                writer.add_scalar("Loss/train", loss, count)
                
            guess_prev=_sample_batched['guess']
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