import segmentation_models as sm
import keras
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import cv2
import numpy as np
import DatasetPrepare
import EdgeDetector
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import Sequence, to_categorical
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
keras.backend.set_image_data_format('channels_last')
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
input_shape = (512, 512, 3)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
def showResult (img, mode=1):
    result =(model.predict (np.expand_dims(cv2.resize (img, (512, 512)), 0))[0]*255).astype (np.uint8)
    #if (mode == 1):
    #    img, edges = EdgeDetector.edgeDetector (img)
    result = cv2.resize (result, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)

    imgBak = np.copy (img)
    
    img[:,:,2] = cv2.addWeighted(result ,0.3,img[:,:,2],0.7,0)
    img = np.clip (img, 0, 255)
    #img = cv2.resize (img, (1920, 1080))
    cv2.imshow ("showResult", img)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result
def predictNum (x_train, num):
    return showResult (x_train [num])
def predict (directory):
    return showResult (cv2.imread (directory))
def startTrain (model):
    model.fit(
       x=x_train,
       y=y_train.astype(np.float32),
       batch_size=2,
       epochs=10,
       validation_data=(x_val, y_val),
    )
    return model

class DataGenerator(Sequence):
    def __init__(self,
                 img_data,
                 labels, 
                 batch_size=32,
                 dim=(512, 512, 3),
                 n_channels=3,
                 n_classes=4,
                 shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.img_data = img_data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.img_indexes = np.arange(len(self.img_data))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        X, y = self.__data_generation(index*self.batch_size, (index+1)*self.batch_size)
        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_data))
        if self.shuffle == True:
            print ("shuffling")
            np.random.shuffle(self.indexes)
    def __data_generation(self, begin, end):
        X = self.img_data [begin:end]
        y = self.labels [begin:end]
        if self.shuffle == True:
            randomNumber = random.randint(0, 10)
            #print (randomNumber)
            #print ((randomNumber % 2))

            #Flip Horizontically
            if ((randomNumber % 2) == 1):
                X = np.flip (X, 2)
                y = np.flip (y, 2)
            #increase brightness
            if (randomNumber > 6):
                gay = np.uint8((randomNumber-5) * 5)
                X[X < 255-gay] += gay
            #decrease brightness
            elif (randomNumber < 4):
                gay = np.uint8((5-randomNumber) * 5)
                X[X > gay] -= gay
            X[0] = rotate (X[0], (randomNumber+1) % 9, preserve_range=True).astype (np.uint8)
            y[0] = rotate (y[0], (randomNumber+1) % 9, preserve_range=True).astype ('float32')
            X[1] = rotate (X[1], (randomNumber+3) % 9, preserve_range=True).astype (np.uint8)
            y[1] = rotate (y[1], (randomNumber+3) % 9, preserve_range=True).astype ('float32')
            X[2] = rotate (X[2], (randomNumber+5) % 9, preserve_range=True).astype (np.uint8)
            y[2] = rotate (y[2], (randomNumber+5) % 9, preserve_range=True).astype ('float32')
            X[3] = rotate (X[3], (randomNumber+7) % 9, preserve_range=True).astype (np.uint8)
            y[3] = rotate (y[3], (randomNumber+7) % 9, preserve_range=True).astype ('float32')
            #randomNumber = 0
        print (X.shape)
        return X, y

#DatasetPrepare || LoadDataset
x, y = DatasetPrepare.LoadDataset("L:\\JAV Folder\\Test Frames")
#x_train, y_train = DatasetPrepare.LoadDataset("L:\\JAV Folder\\testFolderDeleteThis")
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

train_generator = DataGenerator(x_train, y_train.astype('float32'), batch_size = 4, dim = input_shape, n_classes=2, shuffle=True)
val_generator = DataGenerator(x_val, y_val.astype('float32'), batch_size= 4, dim = input_shape,n_classes=2, shuffle=False)

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes = 1, encoder_freeze=True, activation='sigmoid')
#model.load_weights('Augmented model.h5')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

model = startTrain (model)
#model.save("test.h5")

"""gay,gayy = train_generator.__getitem__(1)
for i in range(0,4):
    cv2.imshow("yes", gay[i])
    cv2.imshow("mask", gayy[i])    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
stop
"""
"""
model.fit(
 train_generator,
 steps_per_epoch=len(train_generator),
 epochs=10,
 validation_data=val_generator,
 validation_steps=len(val_generator))"""



predict ("L:/JAV Folder/Test Frames/CAWD-157/frame288000.png")
predict ("L:/JAV Folder/Test Frames/hhd800.com@SORA-343.mp4/frame280800.png")
predict ("L:/JAV Folder/Test Frames/hhd800.com@SORA-343.mp4/frame162000.png")
predict ("L:/JAV Folder/Test Frames/VENU-989/frame114150.png")
predict ("L:/JAV Folder/Test Frames/CAWD-152/frame22050.png")
