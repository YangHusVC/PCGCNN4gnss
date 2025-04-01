########################################################################
# Author(s):    Zhichao Yang
# Date:         
# Desc:                 
########################################################################
import sys, os, csv, datetime
parent_directory = os.path.split(os.getcwd())[0]
src_directory = os.path.join(parent_directory, 'src')
data_directory = os.path.join(parent_directory, 'datasets')
sys.path.insert(0, src_directory)


@hydra.main(config_path="../config", config_name="train_android_conf")
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