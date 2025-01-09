import cv2
import glob
import os
from tqdm import tqdm

train_vol_names = glob.glob(os.path.join('/media/thesis_ssd/data/SCARED/dataset_1/keyframe_3/data/rgb/', '*.png'))
train_vol_names.sort()
for image in tqdm(train_vol_names,total = len(train_vol_names)):
  vol1 = cv2.imread(image, 1)
  vol1 = vol1[0:1024, :, :]
  cv2.imwrite(image, vol1)
 

