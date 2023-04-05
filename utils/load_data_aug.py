import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import random 
from torchvision.transforms import functional as F 
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler


def augment_gaussian_blur(data_sample, sigma_range, per_channel=False, p_per_channel=1):
    data_sample = np.array(data_sample)
    if not per_channel:
        sigma = get_range_val(sigma_range)
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range)
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


# transforms.RandomApply([lambda x: augment_gaussian_blur(x, (0.5, 1.))], p = 0.2), # gaussian blur

class HeartMRIDataset(Dataset):
    def __init__(self, args, images, labels, img_train_trans, img_val_trans, tar_trans, is_train, ood = -1):
        self.image_filenames = images
        self.label_filenames = labels
        self.img_train_trans = img_train_trans
        self.img_val_trans = img_val_trans
        self.tar_trans = tar_trans
        self.train = is_train
        self.ood = ood
        self.args = args
        print("#images:", len((self.image_filenames)), "#labels:", len((self.label_filenames)))
        
    def __getitem__(self, index):
        label_path = self.label_filenames[index]
        image_path = self.image_filenames[index]
        
        # if (os.path.basename(label_path) != os.path.basename(image_path)): 
        #     print("LABEL AND IMAGE DON'T MATCH")
        
        label = np.load(label_path) 
        
        zeros = np.zeros(label.shape)
        ones = np.ones(label.shape)

        label = np.where(label > 0, ones, zeros)

        label = (label*255).astype('uint8')
        label = Image.fromarray(label)

        image = np.load(image_path)
        image = image[None, ...]
        
        # minmax 
        mn = np.min(image)
        mx = np.max(image)
        image = ((image - mn)*255/(mx - mn))[0]
        
        image = image.astype('uint8')
        image = Image.fromarray(image)
    
        if self.train:
            if random.random() > 0.8: 
                rot_degree = np.random.uniform(-10,10) 
                image = F.rotate(image, rot_degree)
                label = F.rotate(label, rot_degree)
            
            if random.random() > 0.8: 
                translate_x = np.random.uniform(-5,5) 
                translate_y = np.random.uniform(-5,5) 
                image = F.affine(image, angle =0, translate = (translate_x,translate_y), scale = 1, shear = 0)
                label = F.affine(label, angle =0, translate = (translate_x,translate_y), scale = 1, shear = 0)
            
            # crop
            image = F.center_crop(image, 128)
            label = F.center_crop(label, 128)

            # minmax 
            image = np.array(image)
            image = image[None, ...]
            mn = np.min(image)
            mx = np.max(image)
            image = ((image - mn)/(mx - mn))[0]

            label = np.array(label)
            label = label / 255.
            
            image = self.img_train_trans(image)
            image = image + self.args.sigma * torch.randn_like(image)
            label = self.tar_trans(label)
        
        else: # if test set 
            # crop
            image = F.center_crop(image, 128)
            label = F.center_crop(label, 128)

            # minmax 
            image = np.array(image)
            image = image[None, ...]
            mn = np.min(image)
            mx = np.max(image)
            image = ((image - mn)/(mx - mn))[0]

            if self.ood ==2 : 
                image = image * 255
                image = image.astype('uint8')
                image = Image.fromarray(image)
                image = self.img_val_trans(image)
                # minmax 
                image = np.array(image)
                image = image[None, ...]
                mn = np.min(image)
                mx = np.max(image)
                image = ((image - mn)/(mx - mn))[0]
                image = self.img_train_trans(image)
            else:
                image = self.img_val_trans(image)

            label = np.array(label)
            label = label / 255.
            
            label = self.tar_trans(label)

        if self.ood == 5 or self.ood == 6: # Gaussian Blur
            image = image.permute(1,2,0)

        patient_id = label_path.split("/")[-1].split(".npy")[0].split("_")[0] 
        filename_only = label_path.split("/")[-1].split(".npy")[0].split("_")[1]
        return {"image": image, "label": label, "patient_id" : patient_id, "file_id": filename_only}

    def __len__(self):
        return len(self.label_filenames)

