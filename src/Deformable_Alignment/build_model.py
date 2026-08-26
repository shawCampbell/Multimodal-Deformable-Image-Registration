from config import Config 

import voxelmorph as vxm
import tensorflow as tf
from tensorflow import keras 

from get_model import get_model

def build_registration_network(config, backbone):
    """Attach a SpatialTransformer to the backbone so the predicted DDF warps the moving
    image and moving label. Returns (registration_model, spatial_transformer)."""
    
    moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
    fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])

    # build transformer layer
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

    # extract the moving/fixed image inputs from the backbone
    moving_image = backbone.input[0]
    fixed_image = backbone.input[1]
    input_moving_label = keras.Input(moving_image_shape, name="moving_label")
    input_fixed_label  = keras.Input(fixed_image_shape,  name="fixed_label")
    inputs = [moving_image, fixed_image, input_moving_label, input_fixed_label]

    # extract ddf
    ddf = backbone.outputs[0]

    # warp the moving image/label with the transformer using the network-predicted ddf
    moved_image = spatial_transformer([moving_image, ddf])
    moved_label = spatial_transformer([input_moving_label, ddf])
    fixed_label = spatial_transformer([input_moving_label, ddf])*0 + input_fixed_label  # hacky: keeps keras quiet about disconnected inputs

    outputs = [moved_image, ddf, moved_label]

    registration_model = keras.Model(inputs=inputs, outputs=outputs)
    return registration_model, spatial_transformer

# if __name__ == '__main__':
#     config = Config()
#     moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
#     fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])

#     registration_model, spatial_transformer = build_registration_network(
#         get_model(moving_image_shape, fixed_image_shape, with_label_inputs=False),
#         moving_image_shape,
#         fixed_image_shape,
#     )

#     print("Registration network:")
#     for tensor in registration_model.inputs:
#         print(f"    in   {tensor.name:16s} {tensor.shape}")
#     for tensor in registration_model.outputs:
#         print(f"    out  {tensor.name:16s} {tensor.shape}")
