from config import Config

import os

import numpy as np
from skimage.transform import resize
import nibabel as nib


def resize_3d_image(image, shape):
    """Resize onto the working grid, then min-max normalise intensities to [0, 1]."""
    resized_image = resize(image, output_shape=shape)
    if np.amax(resized_image) == np.amin(resized_image):
        normalised_image = resized_image
    else:
        normalised_image = (resized_image-np.amin(resized_image))/(np.amax(resized_image)-np.amin(resized_image))
    return normalised_image


def load_dataset_into_cache(f_path, moving_image_shape, fixed_image_shape, with_label_inputs=True):
    """Load every us/mr image (+label) volume below f_path into memory.
    Returns {file_name: {'moving': ..., 'fixed': ...[, 'moving_label': ..., 'fixed_label': ...]}}."""
    moving_images_path = os.path.join(f_path, 'us_images')
    fixed_images_path = os.path.join(f_path, 'mr_images')

    all_names = np.array(os.listdir(fixed_images_path))

    cache = {}

    for f_name in all_names:
        moving_image = nib.load(os.path.join(moving_images_path, f_name)).get_fdata()
        fixed_image = nib.load(os.path.join(fixed_images_path, f_name)).get_fdata()

        moving_image_resized = resize_3d_image(moving_image, moving_image_shape)
        fixed_image_resized = resize_3d_image(fixed_image, fixed_image_shape)

        entry = {
            "moving": moving_image_resized,
            "fixed": fixed_image_resized,
        }

        if with_label_inputs:
            moving_labels_path = os.path.join(f_path, 'us_labels')
            fixed_labels_path = os.path.join(f_path, 'mr_labels')

            moving_label = nib.load(os.path.join(moving_labels_path, f_name)).get_fdata()
            fixed_label = nib.load(os.path.join(fixed_labels_path, f_name)).get_fdata()

            entry["moving_label"] = moving_label
            entry["fixed_label"] = fixed_label

        cache[f_name] = entry

    return cache


if __name__ == '__main__':
    config = Config()
    moving_image_shape = config.config_yaml['moving_image_shape']

    native_volume = np.random.rand(176, 256, 256)  # pretend this came off the scanner at native resolution
    working_volume = resize_3d_image(native_volume, moving_image_shape[:-1])

    print(f"resized {native_volume.shape} -> {working_volume.shape}")
    print(f"intensity range after min-max normalisation: [{working_volume.min():.3f}, {working_volume.max():.3f}]")

    # Once nifti_data/{train,val} exists on disk, load_dataset_into_cache returns e.g.:
    # {
    #     '<case>.nii.gz': {
    #         'moving':       float array, moving_image_shape  (resized + normalised),
    #         'fixed':        float array, fixed_image_shape   (resized + normalised),
    #         'moving_label': native-resolution label volume (..., ..., ..., 6),
    #         'fixed_label':  native-resolution label volume (..., ..., ..., 6),
    #     }, ...
    # }
