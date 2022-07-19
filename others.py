import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import PIL
import imageio

# Function to get the metadata dictionary for the dataset
def get_dict(path, include_augmentations=False):

    # Create dictionary
    d = {
        'ids': 
            {
            'train': [],
            'test': []
            },
        'images': {},
        'masks': {}
        }

    # Images paths
    path_images_train = os.path.join(path, 'training', 'images')
    path_images_test = os.path.join(path, 'test', 'images')
    # Masks path
    path_masks_train = os.path.join(path, 'training', 'groundtruth')

    # Get file names
    filepaths_images_train = get_files(path_images_train)
    filepaths_images_test = get_files(path_images_test)
    filepaths_masks_train = get_files(path_masks_train)

    # Get ids and fill the dictionary
    for filepath in filepaths_images_train:
        id = get_id(filepath)
        d['ids']['train'].append(id)
        d['images'][id] = filepath

    for filepath in filepaths_images_test:
        id = get_id(filepath)
        d['ids']['test'].append(id)
        d['images'][id] = filepath

    for filepath in filepaths_masks_train:
        id = get_id(filepath)
        d['masks'][id] = filepath

    # Also add the augmented training images and masks (if provided)
    if include_augmentations:
        path_images_augmented = os.path.join(path, 'augmentations', 'images')
        path_masks_augmented = os.path.join(path, 'augmentations', 'groundtruth')
        filepaths_images_augmented = get_files(path_images_augmented)
        filepaths_masks_augmented = get_files(path_masks_augmented)
        
        for filepath in filepaths_images_augmented:
            id = get_id(filepath)
            d['ids']['train'].append(id)
            d['images'][id] = filepath

        for filepath in filepaths_masks_augmented:
            id = get_id(filepath)
            d['masks'][id] = filepath

    # Sort the ids
    d['ids']['train'].sort()
    d['ids']['test'].sort()

    return d

# Function to get the paths to all the files in a dictionary
def get_files(path):
    paths = []
    # Iterate over files
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            paths.append(f)

    return paths

# Function to extract the id from a filename
def get_id(filename):
    return int(filename.split('.')[0].split('_')[-1])

# Function to read image and mask (given the paths to them)
def read_image_mask(image_path, mask_path):
    return (imread(image_path) / 255).astype(np.float32), \
           (imread(mask_path, as_gray=True) > 0).astype(np.int8)

# Function to plot the image, predicted mask and ground truth mask (optional)
def plot_image_and_mask(img, mask, ground_truth=None, cmap="Blues"):
    if ground_truth is None:
        fig, axs = plt.subplots(1,2, figsize=(20,10))
        axs[0].imshow(img)
        axs[1].imshow(mask, cmap=cmap)
        plt.show()
    else:
        fig, axs = plt.subplots(1,3, figsize=(20,10))
        axs[0].imshow(img)
        axs[1].imshow(mask, cmap=cmap)
        axs[2].imshow(ground_truth, cmap=cmap)
        plt.show()

# Function to check if a path is creatable
def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

# Function to check if a path exists or is creatable
def is_path_exists_or_creatable(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    '''
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False

# Function to save np array to image
def save_array_as_image(arr, path):
    if not path.endswith('.png'):
        print("[ERROR] Path needs to end with .png")
        return

    if not is_path_creatable(path):
        print("Path is invalid.")
        return

    imageio.imwrite(path, arr)
