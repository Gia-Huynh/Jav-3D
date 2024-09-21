
import cv2
import sys, os
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
ip_size = 256
keras.backend.set_image_data_format('channels_last')
BACKBONE = 'efficientnetb0'
preprocess_input = sm.get_preprocessing(BACKBONE)
input_shape = (ip_size, ip_size, 3)

threshold = 20

model_path = 'Best model.hdf5'
VideoName = "Video/3.mp4"
OutputPath = "VideoOut/3.avi"
def predictAddWeight (img):
    imgResized = np.expand_dims(cv2.resize (img, (ip_size, ip_size)), 0)
    result = model.predict (imgResized)    
    result = (result[0]*255).astype (np.uint8)
    result = cv2.resize (result, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    oldTime = time.time()
    img[:,:,2] = result
    #img[:,:,2] = cv2.addWeighted(result ,0.3,img[:,:,2],0.7,0)
    #img [:,:,2] = img [:,:,2] + result
    #np.add (img[:,:,2], result, out = img[:,:,2])
    #mask = img[:,:,2] > result
    #img[:,:,2] = np.where(mask, img[:,:,2], result) #0.006
    #img.clip(max = 255, out=img) #0.005
    #np.minimum (img, 255, out=img) #0.005
    return img

def predictFUCKME (img):
    return cv2.resize ((model.predict (np.expand_dims(cv2.resize (img, (ip_size, ip_size)), 0))[0]*255).astype (np.uint8), (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)

def predictBatch (img_list):
    return (model.predict (img_list)*255).astype (np.uint8)

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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    while(cap.isOpened()):
      if not(fc < frameCount):
        break
      ret, frame = cap.read()
      

      
      lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
      #lab[...,0] = clahe.apply(lab[...,0])
      buf[fc] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
      #buf[fc] = frame
      if ret == True:
        fc+=1
      else: 
        break
    cap.release()
    return buf, fc, frameRate

if __name__ == '__main__':
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes = 1, encoder_freeze=True, activation='sigmoid')
    #model.load_weights('models/Resnet50_Best Model.hdf5')
    model.load_weights(model_path)
    #model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)
    #result = []
    buf, fc, frameRate = readVideo(VideoName)
    vd_width = len (buf[0][0])
    vd_height = len (buf[0])
    #print (fc)
    #for i in range (0, fc):
        #print (i)
    #    if (i%50 == 0):
    #        print (i)
    #    result [i] = predictFUCKME(buf[i])
    Resized_Image_list = np.empty ((fc, ip_size, ip_size, 3), dtype = np.dtype('uint8'))
    
    for i in range (0, fc):
      Resized_Image_list [i] = cv2.resize(buf[i], (ip_size, ip_size))
      
    result_small = predictBatch (Resized_Image_list)
    
    result = np.empty ((fc, vd_height, vd_width), dtype = np.dtype('uint8'))
    for i in range (0, fc):
      result [i] = cv2.resize(result_small[i], (vd_width, vd_height), interpolation = cv2.INTER_NEAREST)
    heatmapVideo = np.copy (buf)
    heatmapVideo [:,:,:,2] = result
    """
    # B G R thay vi RGB
    PredictMasked = np.copy(buf [:,:,:,0:2])
    PredictMasked [result < threshold] = 0
    #cv2.imshow('Nigger', PredictMasked[5,:,:,0])
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    PredictMasked = np.roll(PredictMasked, 5, axis = 2).astype (np.uint8)
    buf [:,:,:,2][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)] = (0.6 * buf [:,:,:,1][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)] + 0.4 * buf [:,:,:,0][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)]).astype (np.uint8)
    buf [:,:,:,0:2][(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)] = PredictMasked[:,:,:,0:2] [(PredictMasked[:,:,:,0]+PredictMasked[:,:,:,1] > 0)]
    buf [:,:,:,2][result > threshold] = (0.6 * buf [:,:,:,1][result > threshold] + 0.4 * buf [:,:,:,0][result > threshold]).astype (np.uint8)
    buf [:,:,:,0:2][result > threshold] = PredictMasked[:,:,:,0:2] [result > threshold]
    """
    

    #cv2.imshow('PredictMasked', PredictMasked[5])
    #cv2.imshow('result < threshold', (result[5] < threshold).astype (np.uint8) * 255)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

    #print ("Video length: ",fc)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    """for i in range (0, fc):    
        cv2.imshow('Nigger', heatmapVideo[i])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()"""
          
    out = cv2.VideoWriter(OutputPath, fourcc, frameRate, (buf.shape[2], buf.shape[1]))
    for i in range (0, fc):
        out.write (heatmapVideo[i])
    out.release()
    """
    out = cv2.VideoWriter('VideoOut/test deleted.avi', fourcc, frameRate, (buf.shape[2], buf.shape[1]))
    for i in range (0, fc):
        out.write (buf[i])
    out.release()"""
    sys.stdout.flush()
