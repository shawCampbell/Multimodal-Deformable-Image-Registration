from Deformable_Alignment.config import Config

import tensorflow as tf  
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
from tensorflow import keras  
from keras import layers  

def get_model(config, with_label_inputs=True):
    """Residual U-Net: concatenated moving/fixed pair in, dense displacement field (DDF) out."""

    up_filters=config.config_yaml["up_filters"]
    down_filters=config.config_yaml["down_filters"]

    moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
    fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])

    input_moving_image = keras.Input(moving_image_shape)
    input_fixed_image = keras.Input(fixed_image_shape)

    if with_label_inputs:
        input_moving_label = keras.Input(moving_image_shape)
        input_fixed_label = keras.Input(fixed_image_shape)

    concatenate_layer = layers.Concatenate(axis=-1)([input_moving_image, input_fixed_image])

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv3D(32, 3, strides=2, padding="same")(concatenate_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in up_filters:
        # Rediual Block
        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling3D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv3D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in down_filters:
        # Residual Block
        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling3D(2)(x)

        # Project residual
        residual = layers.UpSampling3D(2)(previous_block_activation)
        residual = layers.Conv3D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel displacement prediction layer
    out_ddf = layers.Conv3D(3, 3, activation="linear", padding="same")(x)

    # Define the model
    if with_label_inputs:
        model = keras.Model(inputs=[input_moving_image, input_fixed_image, input_moving_label, input_fixed_label], outputs=[out_ddf])
    else:
        model = keras.Model(inputs=[input_moving_image, input_fixed_image], outputs=[out_ddf])
    return model

if __name__ == '__main__':
    config = Config(file_path='C:\\dev_personal\\Thesis\\config_nfiti_test.yaml')
    moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
    fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])

    backbone = get_model(config, with_label_inputs=False)

    print("Backbone inputs:")
    for tensor in backbone.inputs:
        print(f"    {tensor.name:22s} {tensor.shape}")

    print("\nBackbone output:")
    for tensor in backbone.outputs:
        print(f"    {tensor.name:22s} {tensor.shape}")

    print(f"\nTrainable parameters: {backbone.count_params():,}")