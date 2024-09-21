import os
print(os.getcwd())
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
import keras
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import cv2
import numpy as np
import DatasetPrepare
from sklearn.model_selection import train_test_split
import EdgeDetector
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import Sequence, to_categorical
from skimage.transform import rotate
from sklearn.model_selection import train_test_split

keras.backend.set_image_data_format('channels_last')
BACKBONE = 'resnet152'
preprocess_input = sm.get_preprocessing(BACKBONE)
input_shape = (512, 512, 3)

# load your data
#DatasetPrepare || LoadDataset
#x_rgb, y = DatasetPrepare.LoadDataset("/content/drive/MyDrive/Colab Notebooks")
x_rgb, y = DatasetPrepare.LoadDatasetBinary("L:\\JAV Folder\\Test Frames")
x = x_rgb
print (x.shape)
print (y.shape)
y = y.astype(np.float32)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

class DataGenerator(Sequence):
    def __init__(self,
                 img_data,
                 labels, 
                 batch_size=4,
                 dim=(512, 512, 1),
                 n_channels=3,
                 n_classes=1,
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
        self.indexes = np.arange(len(self.img_data))
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        X, y = self.__data_generation(index*self.batch_size, (index+1)*self.batch_size)
        if (index == self.__len__ ()):
          if self.shuffle == True:
              print ("shuffling")
              np.random.shuffle(self.indexes)

        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass
    def __data_generation(self, begin, end):
        X = self.img_data [begin:end]
        y = self.labels [begin:end]

        
        if self.shuffle == True:
            randomNumber = random.randint(5, 20)
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
            X[0] = rotate (X[0], ((randomNumber+1) % 10-5)*2, preserve_range=True).astype (np.uint8)
            y[0] = rotate (y[0], ((randomNumber+1) % 10-5)*2, preserve_range=True).astype ('float32')
            X[1] = rotate (X[1], ((randomNumber+3) % 10-5)*2, preserve_range=True).astype (np.uint8)
            y[1] = rotate (y[1], ((randomNumber+3) % 10-5)*2, preserve_range=True).astype ('float32')
            X[2] = rotate (X[2], ((randomNumber+5) % 10-5)*2, preserve_range=True).astype (np.uint8)
            y[2] = rotate (y[2], ((randomNumber+5) % 10-5)*2, preserve_range=True).astype ('float32')
            X[3] = rotate (X[3], ((randomNumber+7) % 10-5)*2, preserve_range=True).astype (np.uint8)
            y[3] = rotate (y[3], ((randomNumber+7) % 10-5)*2, preserve_range=True).astype ('float32')
            X[4] = rotate (X[4], ((randomNumber+9) % 10-5)*2, preserve_range=True).astype (np.uint8)
            y[4] = rotate (y[4], ((randomNumber+9) % 10-5)*2, preserve_range=True).astype ('float32')
            #randomNumber = 0
        
        return X, y

train_generator = DataGenerator(x_train, y_train.astype('float32'), batch_size = 7, dim = input_shape, n_classes=1, shuffle=True)
val_generator = DataGenerator(x_val, y_val.astype('float32'), batch_size= 7, dim = input_shape,n_classes=1, shuffle=False)

# define model
model = sm.Unet(BACKBONE, input_shape=(None, None, 3), encoder_weights='imagenet', classes = 1, encoder_freeze=False, activation='sigmoid')
#model = sm.Unet(BACKBONE, input_shape=(None, None, 3), encoder_weights=None, classes = 1, encoder_freeze=False, activation='sigmoid')
#model = tf.keras.models.load_model("testModel")
#model.load_weights('Best Model.hdf5')
model.compile(
    #'adam',
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.4, nesterov=True, name='SGD'),
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="Best Model.hdf5",
    save_weights_only=False,
    monitor='val_iou_score',
    mode='max',
    save_best_only=True)

model.fit(
  train_generator,
  steps_per_epoch=len(train_generator),
  epochs=150,
  validation_data=val_generator,
  validation_steps=len(val_generator),
  callbacks=[model_checkpoint_callback])
print ("Yes")

import shutil
src="Best Model.hdf5"
dst="Best Model_bak.hdf5"
shutil.copy(src,dst)
#!kill $(ps aux | awk '{print $2}')
#!kill -9 -1

print (model.optimizer.weights)
np.save('optimizer.npy', (model.optimizer.get_weights()))
