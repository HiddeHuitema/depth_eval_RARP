import numpy as np
import cv2
import os

depth_dir = ""

names = os.listdir(depth_dir)
# print(names)


for id in range(len(names)):
    # print(name[-2:])
    depth_path = os.path.join(depth_dir,'frame-{:06d}.depth.png'.format(id))
    im = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
    # print(im.shape)
    # print(im)
    np.savez_compressed(os.path.join(depth_dir,'depth_frame-{:06d}.npz'.format(id)),data = im)
    print(id)