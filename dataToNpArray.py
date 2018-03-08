# Importing useful libraries
import os
import numpy as np
from skimage.io import imsave, imread
from glob import glob
import numpy as np
import getopt
import argparse

#getting command line argument
parser = argparse.ArgumentParser(description='paths')
parser.add_argument('--image_path',type=str,help='path of images')
parser.add_argument('--mask_path',type=str,help='path of masks')
parser.add_argument('--save_path',type=str,help='path to save result')
args = parser.parse_args()

# getting images and mask path
path_image=args.image_path
path_mask=args.mask_path
save_result_path=args.save_path 
images = sorted(glob(path_image+'/*.tiff'))
masks = sorted(glob(path_mask+'/*.tiff'))

# Converting images and mask to numpy array
imgs=[]
msks=[]
i=0
for image,mask in zip(images, masks):
    image_mat = imread(image)
    mask_mat = imread(mask)
    imgs.append(image_mat)
    msks.append(mask_mat)

images_np = np.array(imgs)
masks_np=np.array(msks)

# save intermediate result to numpy array
np.save(save_result_path+'/trainingImages.npy', images_np)
np.save(save_result_path+'/trainingMask.npy',masks_np)
