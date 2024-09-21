import cv2
import sys, os, glob
import numpy as np
import DatasetPrepare
import EdgeDetector
import multiprocessing
import matplotlib.pyplot as plt
import random
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from skimage.transform import rotate
from sklearn.model_selection import train_test_split
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)
#mixed_precision.set_global_policy('mixed_float16')
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)

keras.backend.set_image_data_format('channels_last')
BACKBONE = 'resnet152'
preprocess_input = sm.get_preprocessing(BACKBONE)
input_shape = (512, 512, 3)
threshold = 20

model_path = 'Best Model.hdf5'
folderPath= "L:\\jav folder\\Test Frames\\VENU-989"
outputFolderPath = "L:\\jav folder\\Test Frames\\VENU-989\\Result"

def predictFUCKME (img):
    return cv2.resize ((model.predict (np.expand_dims(cv2.resize (img, (512, 512)), 0))[0]*255).astype (np.uint8), (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)

def predictBatch (img_list):
    return (model.predict (img_list)*255).astype (np.uint8)

def readFolder (folderPath, height=512, width=512):
    count = 0
    for Files in glob.glob(folderPath + "\\*.png"):
            count+=1
    ImageList = np.empty ((count, height, width, 3), dtype=np.uint8)
    count = 0
    for Files in glob.glob(folderPath + "\\*.png"):
            Image = cv2.imread(Files)
            ImageList[count] = np.expand_dims(cv2.resize(Image, (width, height)),0)
            count+=1
    NameList = list(map(os.path.basename, glob.glob(folderPath + "\\*.png")))
    return count, NameList, ImageList

if __name__ == '__main__':
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes = 1, encoder_freeze=True, activation='sigmoid')
    model.load_weights(model_path)
    ImageCount, ImageNameList, ImageList = readFolder(folderPath, 1080, 1920)
    
    Resized_Image_list = np.empty ((ImageCount, 512, 512, 3), dtype = np.dtype('uint8'))    
    for i in range (0, ImageCount):
      Resized_Image_list [i] = cv2.resize(ImageList[i], (512, 512))
          
    result_small = predictBatch (Resized_Image_list)
    
    result = np.empty ((ImageCount, 1080, 1920), dtype = np.dtype('uint8'))
    for i in range (0, ImageCount):
      result [i] = cv2.resize(result_small[i], (1920, 1080), interpolation = cv2.INTER_NEAREST)
    heatmapVideo = np.copy (ImageList)
    heatmapVideo [:,:,:,2] = result
    
    for i in range (0, ImageCount):
        cv2.imwrite(os.path.join(outputFolderPath, ImageNameList[i]), heatmapVideo[i])
        
