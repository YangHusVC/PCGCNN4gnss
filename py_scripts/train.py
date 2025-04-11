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
from android_dataset import Android_GNSS_Dataset

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
    test=dataset[0]
    print(test)

if __name__=="__main__":
    main()