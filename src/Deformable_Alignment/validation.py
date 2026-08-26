from Deformable_Alignment.config import Config

import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import voxelmorph as vxm
import neurite as ne

from Deformable_Alignment.get_model import get_model
from Deformable_Alignment.build_model import build_registration_network
from Deformable_Alignment.generators import test_generator

def plot_registration_overview(config, model_save_path, trial,
                               moving_images_val, fixed_images_val, moved_images_val,
                               moving_labels_val, fixed_labels_val, moved_labels_val,
                               ddf_val):
    """Save the qualitative figure pack for the first validation batch of a trial:
    moving/moved/fixed image slices, the same for labels, and the mid-slice flow field."""

    moving_image_shape = tuple(config.config_yaml["moving_image_shape"]) 

    slice_index = moving_image_shape[0]//2

    # VISUALLY CHECK FIXED, MOVING, MOVED IMAGES
    plt.subplot(1, 3, 1)
    plt.imshow(tf.squeeze(moved_images_val[0])[slice_index, :, :], cmap='gray')
    plt.title('Moved Image')

    plt.subplot(1, 3, 2)
    plt.imshow(tf.squeeze(moving_images_val[0])[slice_index, :, :], cmap='gray')
    plt.title('Moving Image')

    plt.subplot(1, 3, 3)
    plt.imshow(tf.squeeze(fixed_images_val[0])[slice_index, :, :], cmap='gray')
    plt.title('Fixed Image')

    save_name_img = os.path.join(model_save_path, f"image_slices_trial_{trial}.png")
    plt.savefig(save_name_img, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    print(f"Saved image slices to {save_name_img}")

    # VISUALLY CHECK LABELS
    plt.subplot(1, 3, 1)
    plt.imshow(tf.squeeze(moved_labels_val[0])[slice_index, :, :], cmap='gray')
    plt.title('Moved Label')

    plt.subplot(1, 3, 2)
    plt.imshow(tf.squeeze(moving_labels_val[0])[slice_index, :, :], cmap='gray')
    plt.title('Moving Label')

    plt.subplot(1, 3, 3)
    plt.imshow(tf.squeeze(fixed_labels_val[0])[slice_index, :, :], cmap='gray')
    plt.title('Fixed Label')

    save_name_lbl = os.path.join(model_save_path, f"label_slices_trial_{trial}.png")
    plt.savefig(save_name_lbl, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    print(f"Saved label slices to {save_name_lbl}")

    # VISUALISE THE PREDICTED DISPLACEMENT FIELD (mid-slice, x/y components only)
    ddf = ddf_val[0].squeeze()           # remove batch/channel dims -> (64, 64, 64, 3)
    mid_plane = ddf[ddf.shape[0] // 2]   # take middle z-slice -> (64, 64, 3)
    flow = mid_plane[::1, ::1]

    save_name_fl = os.path.join(model_save_path, f"flow_slice_{trial}.png")
    fig, _ = ne.plot.flow([flow[..., :2]], width=5, show=False)
    plt.savefig(save_name_fl, dpi=300, bbox_inches='tight')
    plt.close(fig)


def evaluate_validation_dice(config, registration_model, spatial_transformer, 
                             trial=None, model_save_path=None, plot_first_batch=True):
    """Run the deterministic validation generator for all 6 structures, warp the labels with
    the predicted DDFs and collect Dice scores. Optionally render the figure pack for the very
    first batch. Returns the list of per-batch Dice score arrays."""

    dice_scores = []

    for label_num in range(6):
        val_gen = test_generator(config,
                                 4,
                                 start_index=None,
                                 end_index=None,
                                 label_num=label_num,
                                 with_label_inputs=True)

        plotted = False
        while True:
            try:
                (val_inputs, val_outputs) = next(val_gen)
                moving_images_val, fixed_images_val, moving_labels_val, fixed_labels_val = val_inputs
                fixed_images_val, zero_phis_val, fixed_labels_val = val_outputs

                _, ddf_val, _ = registration_model.predict(
                    (moving_images_val, fixed_images_val, moving_labels_val, fixed_labels_val), verbose=0)

                moved_labels_val = spatial_transformer([moving_labels_val, ddf_val])
                moved_images_val = spatial_transformer([moving_images_val, ddf_val])

                if plot_first_batch and label_num == 0 and not plotted:
                    plotted = True
                    plot_registration_overview(config, model_save_path, trial,
                                               moving_images_val, fixed_images_val, moved_images_val,
                                               moving_labels_val, fixed_labels_val, moved_labels_val,
                                               ddf_val)

                dice_score = np.array(-1.0 * vxm.losses.Dice().loss(
                    tf.convert_to_tensor(moved_labels_val, dtype='float32'),
                    tf.convert_to_tensor(fixed_labels_val, dtype='float32')))
                dice_scores.append(dice_score)

                print('.', end='')
            except (IndexError, StopIteration):
                break

    return dice_scores


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

#     smoke_model, smoke_transformer = build_registration_network(
#         get_model(moving_image_shape, fixed_image_shape, with_label_inputs=False),
#         moving_image_shape, fixed_image_shape)

#     smoke_dice = evaluate_validation_dice(smoke_model, smoke_transformer, fake_cache,
#                                         moving_image_shape, fixed_image_shape,
#                                         trial=0, model_save_path='.', plot_first_batch=True)

#     print(f"\n\n{len(smoke_dice)} validation batches scored; mean Dice (untrained net): {np.mean(smoke_dice):.3f}")
