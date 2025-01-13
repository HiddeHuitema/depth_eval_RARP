import cv2
import glob
import os
from tqdm import tqdm

for dataset in range(7,10):
  for keyframe in range(1,10):
    if os.path.isdir(os.path.join('/media/thesis_ssd/data/SCARED/dataset_{}/keyframe_{}/data/rgb/'.format(dataset,keyframe))):
      train_vol_names = glob.glob(os.path.join('/media/thesis_ssd/data/SCARED/dataset_{}/keyframe_{}/data/rgb/'.format(dataset,keyframe), '*.png'))
      train_vol_names.sort()
      for image in tqdm(train_vol_names,total = len(train_vol_names)):
        vol1 = cv2.imread(image, 1)
        vol1 = vol1[0:1024, :, :]
        cv2.imwrite(image, vol1)
 

