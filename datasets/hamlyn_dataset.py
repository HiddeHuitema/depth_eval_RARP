import torch
from torch.utils.data import Dataset
import torchvision


import os
from skimage import io
from PIL import Image


class HamlynDataset(Dataset):
    """The Hamlyn dataset used for evaluating"""

    def __init__(self,rgb_dir,depth_dir,transform_img=None,transform_depth = None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform_img = transform_img
        self.transform_depth = transform_depth

        self.rgb_names = os.listdir(self.rgb_dir)
        self.depth_names = os.listdir(self.depth_dir)


    def __len__(self):
        return len(self.rgb_names)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_name = os.path.join(self.rgb_dir,self.rgb_names[idx])
        depth_name = os.path.join(self.depth_dir,self.depth_names[idx])
        
        rgb_img = torchvision.io.read_image(rgb_name)
        depth_img = Image.open(depth_name)
        # depth_img = torchvision.io.decode_image(depth_file)
        

        if self.transform_img:
            # print('transforming img')
            rgb_img = self.transform_img(rgb_img)
        if self.transform_depth:
            # print('transforming')
            depth_img = self.transform_depth(depth_img)
            
        sample = {'rgb':rgb_img,'depth':depth_img,'id':self.rgb_names[idx]} 

        

        return sample


