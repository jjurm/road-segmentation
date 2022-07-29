import albumentations as A
import utils
import os
from tqdm import tqdm
import numpy as np

def resize(size=384):
    return A.Resize(size, size)

def crop(size):
    return A.Compose([
        A.RandomScale(scale_limit=(-0.1, 0.1)),
        A.CropNonEmptyMaskIfExists(height=size, width=size)
    ])

def pixelwise():
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=0, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=20, contrast_limit=30, brightness_by_max=False, p=0.8),
        A.GaussNoise(var_limit=(10, 50)),
    ])

def affine():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.1, 0.1), rotate_limit=6, p=0.8),
    ])

def distortions():
    return A.Compose([ # Distortions
            A.OneOf([
                A.GridDistortion(num_steps=3, distort_limit=0.03, p=1),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.4, p=1),
                A.ElasticTransform(alpha=10, sigma=20, alpha_affine=3, p=1),                 
            ]),
            A.CenterCrop(390, 390),
            A.Resize(400, 400),
        ], p=0.5)


def aug_with_crop(image_size = 256, crop_prob = 1):
    return A.Compose([
        pixelwise(),
        affine(),
        distortions(),
    ], p = 1)

def aug_without_crop():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=45, p=1),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.Blur(p=0.01, blur_limit = 3),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=50, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=0.6, shift_limit=0.4)                  
        ], p=0.8)
    ], p = 1)


if __name__ == '__main__':
    augmentations_per_sample = 20

    # get data dictionary
    data_path = 'data'
    data = utils.get_dict(data_path, include_augmentations=False)

    # Create augmented directory
    augmented_path = os.path.join(data_path, 'augmentations')
    augmented_images_path = os.path.join(augmented_path, 'images')
    augmented_masks_path = os.path.join(augmented_path, 'groundtruth')

    if os.path.isdir(augmented_path):
        print("[ERROR] Augmentations directory already exists.")
        print("Exiting..")
        exit()
    else:
        os.mkdir(augmented_path)
        os.mkdir(augmented_images_path)
        os.mkdir(augmented_masks_path)

    # Compute the starting index for augmented files
    augmented_idx = max(max(data['ids']['train']), max(data['ids']['test'])) + 1

    # Iterate over training set and augment data
    for idx in tqdm(data['ids']['train']):
        # Get image and mask paths
        img_path = data['images'][idx]
        mask_path = data['masks'][idx]

        # Read the image and the mask
        img, mask = utils.read_image_mask(img_path, mask_path)

        # Create multiple augmentations of the original input and mask
        for i in range(augmentations_per_sample):
            # Augment
            augmented = aug_with_crop(image_size = 352)(image=img, mask=mask)
            img_aug = (augmented['image'] * 255).astype(np.uint8)
            mask_aug = (augmented['mask'] * 255).astype(np.uint8)

            # Save augmentation
            filename = 'satimage_' + str(idx) + '_aug_' + str(augmented_idx) + '.png'
            img_aug_path = os.path.join(augmented_images_path, filename)
            mask_aug_path = os.path.join(augmented_masks_path, filename)

            utils.save_array_as_image(img_aug, img_aug_path)
            utils.save_array_as_image(mask_aug, mask_aug_path)

            # Increment index
            augmented_idx += 1


