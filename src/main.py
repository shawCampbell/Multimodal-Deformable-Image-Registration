from config import Config

if __name__ == "__main__":
    config = Config()

    if config:
        moving_image_shape = tuple(config.config_yaml["moving_image_shape"]) 
        fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"]) 

        print(f"moving (US) volume shape : {moving_image_shape}")
        print(f"fixed  (MR) volume shape : {fixed_image_shape}")
        print(f"the network predicts one 3-vector of displacements per fixed voxel -> a DDF of shape {fixed_image_shape[:-1] + (3,)}")