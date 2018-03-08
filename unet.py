#importing python libraries
from skimage.transform import resize
from skimage.io import imsave,imread
import numpy as np
from matplotlib import pyplot as plt
import argparse

#importing keras function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

#import our model
from unet_model import unet

#getting command line argument
parser = argparse.ArgumentParser(description='paths')
parser.add_argument('--mode',type=str,help='train or predict?')
parser.add_argument('--image_path',type=str,help='path of images')
parser.add_argument('--mask_path',type=str,help='path of masks')
parser.add_argument('--save_path',type=str,help='path to save result')
args = parser.parse_args()

# getting images and mask path
path_image=args.image_path
mode =args.mode
if (mode=="train"):
    path_mask=args.mask_path
save_result_path=args.save_path 

#Train method to train Unet model on images and given mask
def train():
    #loading images and mask a np array
    images = np.load(path_image)
    mask = np.load(path_mask)
    
    print("-------data loaded---------")

    #reshaping training images and mask to 4 channels
    reshaped=[]
    for image in images:
        reshaped.append(resize(image,(512,512),preserve_range=True))
    reshaped= np.array(reshaped)
    train = reshaped[...,np.newaxis]

    print("-------images reshaped---------")

    reshaped_mask=[]
    for msk in mask:
        reshaped_mask.append(resize(msk,(512,512),preserve_range=True))
    reshaped_mask= np.array(reshaped_mask)
    masks = reshaped_mask[...,np.newaxis]

    print("-------mask reshaped---------")

    #augmenting data
    train = train.astype('float32')
    mask_images = masks.astype('float32')
    training_images = train - np.mean(train)
    training_images = training_images / np.std(train)
    mask_images = mask_images/255 

    print("-------data normalized---------")
    print("-------loading model---------")
    model = unet()

    #Fitting and saving model
    model.fit(training_images, mask_images, batch_size=32, epochs=30, verbose=1, shuffle=True)
    model.save(args.save_path+"/model.h5")

#predict method to use saved model to predict mask of test images
def predict():
    
    #loading and reshaping image
    images = np.load(path_image)
    reshaped=[]
    for image in images:
        reshaped.append(resize(image,(512,512),preserve_range=True))
    reshaped= np.array(reshaped)
    test = reshaped[...,np.newaxis]
    
    #Augmenting test data
    test = test.astype('float32')
    testing_images = test - np.mean(test)
    testing_images = testing_images / np.std(test)
    
    #loading model and predicting mask

    model=unet()
    model.load_weights('model.h5')

    predicted = model.predict(testing_images, verbose=1)
    np.save(save_result_path+'/predictedMask.npy', predicted)

if (mode=="train"):
    train()
if(mode=="predict"):
    predict()
