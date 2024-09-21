
import cv2
import sys, os
import numpy as np
import DatasetPrepare
import EdgeDetector
import multiprocessing
import matplotlib.pyplot as plt
import random
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from skimage.transform import rotate
from sklearn.model_selection import train_test_split
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import segmentation_models as sm
import keras
import tensorflow as tf


from tensorflow.keras.utils import Sequence, to_categorical

from tensorflow.keras import mixed_precision

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
"""
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)"""

keras.backend.set_image_data_format('channels_last')
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)
input_shape = (512, 512, 1)
threshold = 50
def predictAddWeight (img):
    
    imgResized = np.expand_dims(cv2.resize (img, (512, 512)), 0)
    
    result = model.predict (imgResized)

    
    result = (result[0]*255).astype (np.uint8)
    result = cv2.resize (result, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    print (time.time() - oldTime)
    oldTime = time.time()
    img[:,:,2] = result
    return img

def predictFUCKME (img):
    return cv2.resize ((model.predict (np.expand_dims(cv2.resize (img, (512, 512)), 0))[0]*255).astype (np.uint8), (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)

def predictBatch (img_list, batch_size = 1):
    return (model.predict (img_list, batch_size=batch_size)*255).astype (np.uint8)

def readVideo (videoName):
    global result
    cap = cv2.VideoCapture(videoName)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameRate = int (cap.get(cv2.CAP_PROP_FPS))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    result = np.empty ((frameCount, frameHeight, frameWidth), np.dtype ('uint8'))
    if (cap.isOpened()== False): 
      print("Error opening video  file")
    fc = 0
    ret = True
    while(cap.isOpened()):
      if not(fc < frameCount):
        break
      ret, buf[fc] = cap.read()
      if ret == True:
        fc+=1
      else: 
        break
    cap.release()
    return buf, fc, frameRate

if __name__ == '__main__':
    model = sm.Unet(BACKBONE, input_shape=(None, None, 3), encoder_weights='imagenet', classes = 1, activation='sigmoid')
    model.load_weights('models/Best Model Binary.hdf5')
    #model.load_weights('models/Bad Model Binary.hdf5')
    #model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)
    #result = []
    buf, fc, frameRate = readVideo("2.mp4")
    #print (fc)
    #for i in range (0, fc):
        #print (i)
    #    if (i%50 == 0):
    #        print (i)
    #    result [i] = predictFUCKME(buf[i])
    Resized_Image_list = np.empty ((fc, 512, 512, 3), dtype = np.dtype('uint8'))
    
    buf_binary = (buf[:,:,:,0] * 0.114 + buf[:,:,:,1] * 0.587 + buf[:,:,:,2] * 0.299).astype (np.uint8)
    buf_binary = np.broadcast_to(buf_binary[...,None], buf_binary.shape+(3,))
    
    for i in range (0, fc):
      Resized_Image_list [i] = cv2.resize(buf_binary[i], (512, 512))

    result_small = predictBatch (Resized_Image_list, 1)
    
    result = np.empty ((fc, 1080, 1920), dtype = np.dtype('uint8'))
    for i in range (0, fc):
      result [i] = cv2.resize(result_small[i], (1920, 1080)) #, interpolation = cv2.INTER_NEAREST
    heatmapVideo = np.copy (buf)
    heatmapVideo [:,:,:,2] = result
    
    # B G R thay vi RGB
    PredictMasked = np.copy(buf [:,:,:,0:2])
    PredictMasked [result < threshold] = 0
    #cv2.imshow('Nigger', PredictMasked[5,:,:,0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    PredictMasked = np.roll(PredictMasked, 15, axis = 2).astype (np.uint8)
    buf [:,:,:,2][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)] = (0.6 * buf [:,:,:,1][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)] + 0.4 * buf [:,:,:,0][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)]).astype (np.uint8)
    buf [:,:,:,0:2][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)] = PredictMasked[:,:,:,0:2] [(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)]
    buf [:,:,:,2][result > threshold] = (0.6 * buf [:,:,:,1][result > threshold] + 0.4 * buf [:,:,:,0][result > threshold]).astype (np.uint8)
    buf [:,:,:,0:2][result > threshold] = PredictMasked[:,:,:,0:2] [result > threshold]
    
    


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
          
    out = cv2.VideoWriter('test Binary 2_1.avi', fourcc, frameRate, (buf.shape[2], buf.shape[1]))
    for i in range (0, fc):
        out.write (heatmapVideo[i])
    out.release()
    
    out = cv2.VideoWriter('test Binary_1.avi', fourcc, frameRate, (buf.shape[2], buf.shape[1]))
    for i in range (0, fc):
        out.write (buf[i])
    out.release()
    sys.stdout.flush()
