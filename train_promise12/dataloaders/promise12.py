import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from skimage.exposure import equalize_adapthist
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
import itertools
import torchvision
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

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

def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=1, pixel_level=True):
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

# New augmentation functions
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

class Promise12(Dataset):

    def __init__(self, data_dir, mode, out_size):
        # store data in the npy file
        self.out_size = out_size
        np_data_path = os.path.join(data_dir, 'npy_image')
        if not os.path.exists(np_data_path):
            os.makedirs(np_data_path)
            data_to_array(data_dir, np_data_path, self.out_size, self.out_size)
        else:
            print('read the data from: {}'.format(np_data_path))
        self.data_dir = data_dir
        self.sample_list = []
        self.mode = mode
        # read the data from npy
        if self.mode == 'train':
            self.X_train = np.load(os.path.join(np_data_path, 'X_train.npy'))
            self.y_train = np.load(os.path.join(np_data_path, 'y_train.npy'))
        elif self.mode == "val":
            with open(data_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
          # self.X_val = np.load(os.path.join(np_data_path, 'X_val.npy'))
          # self.y_val = np.load(os.path.join(np_data_path, 'y_val.npy'))
        elif self.mode == 'test':
            with open(data_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        
    def __getitem__(self, i):
        np_data_path = os.path.join(self.data_dir, 'npy_image')
        
        if self.mode == 'train':
            img, mask = self.X_train[i], self.y_train[i]  # [224,224] [224,224]
            
            # Apply comprehensive augmentation instead of basic transforms
            img, mask = comprehensive_augmentation(img, mask)

            image_strong, label_strong = cutout_gray(img,mask,p=0.5)
            image_strong = color_jitter(img).type("torch.FloatTensor")
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask)
            mask_strong = torch.from_numpy(label_strong)
            sample = {"image": img_tensor, 
                      "image_strong": image_strong,
                      "mask": mask_tensor, 
                      'mask_strong': mask_strong,
                      }

        elif self.mode == 'val':
            case = self.sample_list[i]
            max_retries = 10  # Maximum number of retries to find a valid file
            
            for attempt in range(max_retries):
                try:
                    img = np.load(os.path.join(np_data_path, '{}.npy'.format(case)))
                    mask = np.load(os.path.join(np_data_path, '{}_segmentation.npy'.format(case)))
                    img_tensor = torch.from_numpy(img)
                    mask_tensor = torch.from_numpy(mask)
                    sample = {"image": img_tensor, "mask": mask_tensor}
                    break
                    
                except FileNotFoundError:
                    # print(f"Warning: File not found for case {case}, trying next available file...")
                    # Try the next index
                    i = (i + 1) % len(self.sample_list)
                    case = self.sample_list[i]
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Could not find any valid files after {max_retries} attempts")
                except Exception as e:
                    print(f"Error loading case {case}: {e}, trying next available file...")
                    # Try the next index
                    i = (i + 1) % len(self.sample_list)
                    case = self.sample_list[i]
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Could not find any valid files after {max_retries} attempts")
                        
        elif self.mode == 'test':
            case = self.sample_list[i]
            max_retries = 10  # Maximum number of retries to find a valid file
            
            for attempt in range(max_retries):
                try:
                    img = np.load(os.path.join(np_data_path, '{}.npy'.format(case)))
                    mask = np.load(os.path.join(np_data_path, '{}_segmentation.npy'.format(case)))
                    img_tensor = torch.from_numpy(img)
                    mask_tensor = torch.from_numpy(mask)
                    sample = {"image": img_tensor, "mask": mask_tensor}
                    break
                    
                except FileNotFoundError:
                    # print(f"Warning: File not found for case {case}, trying next available file...")
                    # Try the next index
                    i = (i + 1) % len(self.sample_list)
                    case = self.sample_list[i]
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Could not find any valid files after {max_retries} attempts")
                except Exception as e:
                    print(f"Error loading case {case}: {e}, trying next available file...")
                    # Try the next index
                    i = (i + 1) % len(self.sample_list)
                    case = self.sample_list[i]
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Could not find any valid files after {max_retries} attempts")
                        
        return sample

    def __len__(self):
        if self.mode == 'train':
            return self.X_train.shape[0]
        elif self.mode == 'val':
            return len(self.sample_list)
        elif self.mode == 'test':
            return len(self.sample_list)


def data_to_array(base_path, store_path, img_rows, img_cols):
    global min_val, max_val
    fileList = os.listdir(base_path)
    fileList = sorted((x for x in fileList if '.mhd' in x))

    val_list = [35, 36, 37, 38, 39]
    test_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    
    train_list = list(set(range(50)) - set(val_list) - set(test_list))

    for the_list in [train_list]:
        images = []
        masks = []

        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append(imgs)
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs_norm = np.zeros([len(imgs), img_rows, img_cols])
                for mm, img in enumerate(imgs):
                    min_val = np.min(img)  # Min-Max归一化
                    max_val = np.max(img)
                    imgs_norm[mm] = (img - min_val) / (max_val - min_val)
                images.append(imgs_norm)

        # images: slices x w x h ==> total number x w x h
        images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols)  # (1250,256,256)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
        masks = masks.astype(np.uint8)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)
        images = images.astype(np.float32)

        np.save(os.path.join(store_path, 'X_train.npy'), images)
        np.save(os.path.join(store_path, 'y_train.npy'), masks)
    for the_list in [val_list, test_list]:
        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs = imgs.astype(np.uint8)
                np.save(os.path.join(store_path, '{}.npy'.format(filename[:-4])), imgs)
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs_norm = np.zeros([len(imgs), img_rows, img_cols])
                for mm, img in enumerate(imgs):
                    min_val = np.min(img)  # Min-Max归一化
                    max_val = np.max(img)
                    imgs_norm[mm] = (img - min_val) / (max_val - min_val)
                images = smooth_images(imgs_norm)
                images = images.astype(np.float32)
                np.save(os.path.join(store_path, '{}.npy'.format(filename[:-4])), images)



def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs


def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                 timeStep=t_step,
                                 numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)

    return imgs