from Deformable_Alignment.config import Config

import os

from Deformable_Alignment.caching import load_dataset_into_cache
from Deformable_Alignment.training import train_model

if __name__ == "__main__":
    config = Config('C:\\dev_personal\\Thesis\\config_nfiti_preproc.yaml')

    moving_image_shape = tuple(config.config_yaml["moving_image_shape"]) 
    fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"]) 
    train_path = config.config_yaml["train_path"]
    val_path = config.config_yaml["val_path"]

    print(f"moving (US) volume shape : {moving_image_shape}")
    print(f"fixed  (MR) volume shape : {fixed_image_shape}")
    print(f"the network predicts one 3-vector of displacements per fixed voxel -> a DDF of shape {fixed_image_shape[:-1] + (3,)}")

    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    config.train_cache = load_dataset_into_cache(config, train_path, with_label_inputs=True)
    config.test_cache = load_dataset_into_cache(config, val_path, with_label_inputs=True)

    intensity_metric = 'NCC'
    weakly_supervised = False
    Verbose=False

    train_model(config, intensity_metric, weakly_supervised, experiment_name='run')
