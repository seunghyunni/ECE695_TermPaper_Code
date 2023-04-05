import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MinMaxScaler


class HeartMRIDataset(Dataset):
    def __init__(self, images, labels, img_trans, tar_trans):
        self.image_filenames = images
        self.label_filenames = labels
        self.img_trans = img_trans
        self.tar_trans = tar_trans

        print("#images:", len((self.image_filenames)), "#labels:", len((self.label_filenames)))
        
    def __getitem__(self, index):
        label_path = self.label_filenames[index]
        image_path = self.image_filenames[index]
        
        if (os.path.basename(label_path) != os.path.basename(image_path)): 
            print("LABEL AND IMAGE DON'T MATCH")
        
        label = np.load(label_path)
        # label[label>0] = 1
        # label = Image.fromarray(np.uint8(label*255))
        label = self.tar_trans(label)
        #label = label > 0.5
        
        label_filename = label_path.split("/")[-1]
        
        scaler = MinMaxScaler(feature_range=(0,1))

        image = np.load(image_path)
        # scaler.fit(image)
        # image = scaler.transform(image)
        #print(np.unique(image))
        image = Image.fromarray(np.uint8(image*255))
        image = self.img_trans(image)

        patient_id = label_filename.split(".npy")[0].split("_")[0] 
        filename_only = label_filename.split(".npy")[0].split("_")[1] 
        return {"image": image, "label": label, "patient_id" : patient_id, "file_id": filename_only}

    def __len__(self):
        return len(self.label_filenames)

