import os
# import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter



def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def blur(image, p=0.5):
    if random.random() < p:
        max = np.max(image)
        min = np.min(image)
        image = (image - min) / (max - min)
        image = gaussian_filter(image, sigma=1)
        image = image * (max - min) + min
    return image

# New comprehensive augmentation functions
def adjust_brightness_contrast(image, brightness_factor=0.2, contrast_factor=0.2):
    """Adjust brightness and contrast of the image"""
    if random.random() < 0.5:
        # Brightness adjustment
        brightness_delta = random.uniform(-brightness_factor, brightness_factor)
        image = np.clip(image + brightness_delta, 0, 1)
    
    if random.random() < 0.5:
        # Contrast adjustment
        contrast_factor = random.uniform(1 - contrast_factor, 1 + contrast_factor)
        mean_val = np.mean(image)
        image = np.clip((image - mean_val) * contrast_factor + mean_val, 0, 1)
    
    return image

def add_gaussian_noise(image, std=0.05):
    """Add Gaussian noise to the image"""
    if random.random() < 0.3:
        noise = np.random.normal(0, std, image.shape)
        image = np.clip(image + noise, 0, 1)
    return image

def elastic_deformation(image, label, alpha=1, sigma=50, alpha_affine=50):
    """Apply elastic deformation to image and label"""
    if random.random() < 0.3:
        shape = image.shape
        shape_size = shape[:2]
        
        # Random affine transformation
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        label = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        
        # Elastic deformation
        dx = gaussian_filter((np.random.rand(*shape_size) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape_size) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape_size)
        label = map_coordinates(label, indices, order=1, mode='reflect').reshape(shape_size)
    
    return image, label

def random_crop_and_resize(image, label, crop_ratio=0.8):
    """Random crop and resize back to original size"""
    if random.random() < 0.3:
        h, w = image.shape
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        # Random crop
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        image_crop = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        label_crop = label[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Resize back to original size
        image = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return image, label

def random_blur(image, kernel_size=3):
    """Apply random Gaussian blur"""
    if random.random() < 0.2:
        sigma = random.uniform(0.1, 1.0)
        image = gaussian_filter(image, sigma=sigma)
    return image

def comprehensive_augmentation(image, label):
    """Apply comprehensive data augmentation pipeline"""
    # Apply all augmentations with different probabilities
    if random.random() < 0.7:  # 70% chance for rotation/flip
        image, label = random_rot_flip(image, label)
    
    if random.random() < 0.5:  # 50% chance for rotation
        image, label = random_rotate(image, label)
    
    if random.random() < 0.4:  # 40% chance for elastic deformation
        image, label = elastic_deformation(image, label)
    
    if random.random() < 0.4:  # 40% chance for crop and resize
        image, label = random_crop_and_resize(image, label)
    
    # Photometric augmentations
    image = adjust_brightness_contrast(image)
    image = add_gaussian_noise(image)
    image = random_blur(image)
    
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3, value_min=0, value_max=1,
                pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

    return img, mask


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        
        # Apply comprehensive augmentation
        image, label = comprehensive_augmentation(image, label)
        
        # Resize to output size
        image = self.resize(image)
        label = self.resize(label)
        
        sample = {"image": image, "label": label}
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # weak augmentation is rotation / flip
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # strong augmentation is color jitter
        image_strong, label_strong = cutout_gray(image, label, p=0.5)
        image_strong = color_jitter(image_strong).type("torch.FloatTensor")
        # image_strong = blur(image, p=0.5)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.uint8))
        label_strong = torch.from_numpy(label_strong.astype(np.uint8))
        sample = {
            "image": image,
            "image_strong": image_strong,
            "label": label,
            "label_strong": label_strong
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


