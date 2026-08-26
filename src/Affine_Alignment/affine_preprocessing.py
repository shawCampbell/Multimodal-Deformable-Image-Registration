import os
import nibabel as nib
import numpy as np
import ants

def collapse_labels(lbl):
    """
    Collapse one-hot/multi-channel labels into a single label map.
    lbl shape: (H, W, D, C)
    returns: (H, W, D) with values 0..C
    """
    return np.argmax(lbl, axis=-1)

def preprocess_affine_registration(data_dir, output_dir):
    """
    Preprocess dataset by affinely registering US to MR using all labels.

    Args:
        data_dir (str): input directory containing 'mr_images', 'mr_labels',
                        'us_images', 'us_labels'.
        output_dir (str): output directory to save registered volumes.
    Returns:
        transforms_list: list of fwdtransforms (length = number of cases).
    """

    mr_img_dir = os.path.join(data_dir, "mr_images")
    mr_lbl_dir = os.path.join(data_dir, "mr_labels")
    us_img_dir = os.path.join(data_dir, "us_images")
    us_lbl_dir = os.path.join(data_dir, "us_labels")

    # Create output directories
    for sub in ["mr_images", "mr_labels", "us_images", "us_labels"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    case_names = sorted(os.listdir(mr_img_dir))
    transforms_list = []

    for case in case_names:
        try:
            print(f"Processing {case}...")

            # Load NIfTI files
            mr_img_nib = nib.load(os.path.join(mr_img_dir, case))
            mr_lbl_nib = nib.load(os.path.join(mr_lbl_dir, case))
            us_img_nib = nib.load(os.path.join(us_img_dir, case))
            us_lbl_nib = nib.load(os.path.join(us_lbl_dir, case))

            mr_img = mr_img_nib.get_fdata()
            mr_lbl = mr_lbl_nib.get_fdata()  # shape: H x W x D x C
            us_img = us_img_nib.get_fdata()
            us_lbl = us_lbl_nib.get_fdata()  # shape: H x W x D x C

            # Collapse all label channels into single label maps for registration
            mr_lbl_collapse = collapse_labels(mr_lbl)
            us_lbl_collapse = collapse_labels(us_lbl)

            # Convert to ANTs images
            mr_lbl_img = ants.from_numpy(mr_lbl_collapse.astype(np.float32))
            us_lbl_img = ants.from_numpy(us_lbl_collapse.astype(np.float32))

            # Register based on all labels → affine
            reg = ants.registration(
                fixed=mr_lbl_img,
                moving=us_lbl_img,
                type_of_transform='Affine',
                interpolator='nearestNeighbor'
            )
            transforms_list.append(reg['fwdtransforms'])

            # Apply transform to US image (linear interpolator)
            us_img_ants = ants.from_numpy(us_img.astype(np.float32))
            moved_us_img = ants.apply_transforms(
                fixed=ants.from_numpy(mr_img.astype(np.float32)),
                moving=us_img_ants,
                transformlist=reg['fwdtransforms'],
                interpolator='linear'
            )

            # Apply transform to US label channels (nearest-neighbor)
            moved_channels = []
            for c in range(us_lbl.shape[-1]):
                mov_lbl_np = np.squeeze(us_lbl[..., c])
                mov_lbl_img = ants.from_numpy(mov_lbl_np.astype(np.float32))
                moved_lbl_img = ants.apply_transforms(
                    fixed=ants.from_numpy(mr_lbl[..., c].astype(np.float32)),
                    moving=mov_lbl_img,
                    transformlist=reg['fwdtransforms'],
                    interpolator='nearestNeighbor'
                )
                moved_channels.append(moved_lbl_img.numpy())
            moved_us_lbl = np.stack(moved_channels, axis=-1)

            # Save registered outputs (keep multi-channel format for labels)
            nib.save(mr_img_nib, os.path.join(output_dir, "mr_images", case))
            nib.save(mr_lbl_nib, os.path.join(output_dir, "mr_labels", case))
            nib.save(nib.Nifti1Image(moved_us_img.numpy(), mr_img_nib.affine),
                     os.path.join(output_dir, "us_images", case))
            nib.save(nib.Nifti1Image(moved_us_lbl, mr_img_nib.affine),
                     os.path.join(output_dir, "us_labels", case))

        except Exception as e:
            print(f"⚠️ Registration failed for {case}: {e}")
            # Save originals unchanged
            nib.save(mr_img_nib, os.path.join(output_dir, "mr_images", case))
            nib.save(mr_lbl_nib, os.path.join(output_dir, "mr_labels", case))
            nib.save(us_img_nib, os.path.join(output_dir, "us_images", case))
            nib.save(us_lbl_nib, os.path.join(output_dir, "us_labels", case))

    print("Preprocessing complete. Registered (or original) data saved to:", output_dir)
    return transforms_list

if __name__ == '__main__':
    # data_dir = "C:\\dev_personal\\Thesis\\train\\train"
    # output_dir = "C:\\dev_personal\\Thesis\\src\\Affine_Alignment\\preproc_data_train\\train"
    # preprocess_affine_registration(data_dir, output_dir)

    data_dir = "C:\\dev_personal\\Thesis\\val\\val"
    output_dir = "C:\\dev_personal\\Thesis\\src\\Affine_Alignment\\preproc_data_val\\val"
    preprocess_affine_registration(data_dir, output_dir)