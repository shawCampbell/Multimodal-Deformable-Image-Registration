from Deformable_Alignment.config import Config

import numpy as np
from Deformable_Alignment.caching import resize_3d_image

def train_generator_(config, batch_size, with_label_inputs=True):
    """Infinite shuffled batches. y_true = [fixed image, zero DDF (regularisation target), fixed label]."""

    moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
    fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])
    cache = config.train_cache

    all_names = list(cache.keys())

    while True:
        batch_names = np.random.permutation(all_names)[:batch_size]

        moving_images_batch = np.zeros((batch_size, *moving_image_shape))
        fixed_images_batch = np.zeros((batch_size, *fixed_image_shape))

        if with_label_inputs:
            moving_labels_batch = np.zeros((batch_size, *moving_image_shape))
            fixed_labels_batch = np.zeros((batch_size, *fixed_image_shape))

        for i, f_name in enumerate(batch_names):
            entry = cache[f_name]
            moving = entry["moving"]
            fixed  = entry["fixed"]

            if with_label_inputs:
                label_to_select = np.random.randint(6)
                moving_label = resize_3d_image(entry["moving_label"][:, :, :, label_to_select], moving_image_shape)
                fixed_label  = resize_3d_image(entry["fixed_label"][:, :, :, label_to_select], fixed_image_shape)
            else:
                moving_label, fixed_label = None, None

            # assign into batch
            moving_images_batch[i] = moving
            fixed_images_batch[i]  = fixed
            if with_label_inputs:
                moving_labels_batch[i] = moving_label
                fixed_labels_batch[i]  = fixed_label

        zero_phis = np.zeros([batch_size, *moving_image_shape[:-1], 3])

        if with_label_inputs:
            inputs = (moving_images_batch, fixed_images_batch, moving_labels_batch, fixed_labels_batch)
            outputs = (fixed_images_batch, zero_phis, fixed_labels_batch)
        else:
            inputs = (moving_images_batch, fixed_images_batch)
            outputs = (fixed_images_batch, zero_phis)

        yield inputs, outputs



def test_generator(config, batch_size, start_index, end_index, label_num, with_label_inputs=True):
    """Deterministic walk over cache[start:end] for one label structure."""

    moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
    fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])
    cache = config.test_cache

    all_names = list(cache.keys())[start_index:end_index]
    n_steps = int(np.floor(len(all_names) / batch_size))

    for step in range(n_steps):
        batch_names = all_names[step*batch_size:(step+1)*batch_size]

        moving_images_batch = np.zeros((batch_size, *moving_image_shape))
        fixed_images_batch = np.zeros((batch_size, *fixed_image_shape))

        if with_label_inputs:
            moving_labels_batch = np.zeros((batch_size, *moving_image_shape))
            fixed_labels_batch = np.zeros((batch_size, *fixed_image_shape))

        for i, f_name in enumerate(batch_names):
            entry = cache[f_name]
            moving_images_batch[i] = entry["moving"]
            fixed_images_batch[i] = entry["fixed"]

            if with_label_inputs:
                moving_labels_batch[i] = resize_3d_image(entry["moving_label"][:, :, :, label_num], moving_image_shape)
                fixed_labels_batch[i] = resize_3d_image(entry["fixed_label"][:, :, :, label_num], fixed_image_shape)

        zero_phis = np.zeros([batch_size, *moving_image_shape[:-1], 3])

        if with_label_inputs:
            inputs = [moving_images_batch, fixed_images_batch, moving_labels_batch, fixed_labels_batch]
            outputs = [fixed_images_batch, zero_phis, fixed_labels_batch]
        else:
            inputs = [moving_images_batch, fixed_images_batch]
            outputs = [fixed_images_batch, zero_phis]

        yield inputs, outputs

# if __name__ == '__main__':
#     config = Config()
#     moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
#     fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])

#     fake_cache = {
#         f"case_{i:02d}.nii.gz": {
#             "moving":       np.random.rand(*moving_image_shape).astype("float32"),
#             "fixed":        np.random.rand(*fixed_image_shape).astype("float32"),
#             # native-resolution labels holding 6 structures each, like the real NIfTI data
#             "moving_label": np.random.randint(0, 2, (96, 96, 96, 6)).astype("float32"),
#             "fixed_label":  np.random.randint(0, 2, (96, 96, 96, 6)).astype("float32"),
#         }
#         for i in range(8)
#     }

#     (inputs, outputs) = next(train_generator_(fake_cache, 4, moving_image_shape, fixed_image_shape))

#     print("one TRAIN batch:")
#     for label_name, arr in zip(["moving image", "fixed image", "moving label", "fixed label"], inputs):
#         print(f"    X  {label_name:14s} {arr.shape}")
#     print(f"    y  fixed image    {outputs[0].shape}  -> similarity head target")
#     print(f"    y  zero phi       {outputs[1].shape} -> regularisation head target (all zeros)")
#     print(f"    y  fixed label    {outputs[2].shape}  -> Dice head target")

#     (val_inputs, val_outputs) = next(test_generator(fake_cache, 4, moving_image_shape, fixed_image_shape,
#                                                     start_index=0, end_index=4, label_num=0, with_label_inputs=True))
#     print("\none TEST batch (deterministic, structure 0):", ", ".join(str(a.shape) for a in val_inputs))
