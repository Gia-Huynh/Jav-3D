import os
print(os.getcwd())
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
#import keras
import tensorflow as tf
from tensorflow import keras
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
  
halfPrecision = 0

if (halfPrecision == 1):
  from tensorflow.keras import layers
  from tensorflow.keras import mixed_precision
  mixed_precision.set_global_policy('mixed_float16')
  floatType = 'float16'
  epsilon = 1e-04
else:
  floatType = 'float32'
  epsilon = 1e-06
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
input_shape = (256, 256, 3)

Load_opt = 0
Load_weight = 0
encoder_freeze = True
epochs = 100
optimizer_file_name = 'optimizer.npy'
# load your data
#DatasetPrepare || LoadDataset
x, y = DatasetPrepare.LoadDataset("L:\\JAV Folder\\Test Frames")
y = y.astype(np.float32)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

class DataGenerator(Sequence):
    def __init__(self,
                 img_data,
                 labels, 
                 batch_size=16,
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
        #self.img_indexes = np.arange(len(self.img_data))
        self.on_epoch_end()
        self.indexes = np.arange(len(self.img_data))

        self.mean = 0
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        X, y = self.__data_generation(index*self.batch_size, (index+1)*self.batch_size)
        if (index+1 == self.__len__ ()):
          if self.shuffle == True:
              #print (self.indexes[0:2])
              np.random.shuffle(self.indexes)
          #else:
              #print ("validating")

        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass
    def __data_generation(self, begin, end):
        X = np.empty((self.batch_size, *self.dim), dtype = self.img_data.dtype)
        y = np.empty(((self.labels [begin:end]).shape), dtype = floatType) #self.labels.dtype
        for i, Id in enumerate(self.indexes[begin:end]):
          X[i,] = np.ndarray.copy(self.img_data [Id])
          y[i,] = np.ndarray.copy(self.labels [Id])
        if self.shuffle == True:
            #0,1,...,7,8,9,10
            randomNumber = random.randint(0, 10)
            #Flip Horizontically
            if ((randomNumber % 2) == 1):
                X = np.flip (X, 2)
                y = np.flip (y, 2)
            #increase brightness
            if (randomNumber > 7):
                gay = np.uint8((randomNumber-7) * 4)
                X[X < 255-gay] += gay
                X[X > 255-gay] = 255
            #decrease brightness
            elif (randomNumber < 5):
                gay = np.uint8((randomNumber+1) * 5)
                X[X > gay] -= gay
                X[X < gay] = 0
            if ((randomNumber > 9) or (randomNumber <1)or (randomNumber==5)or (randomNumber==6)):
              clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
              for pic in X:
                  lab = cv2.cvtColor(pic, cv2.COLOR_BGR2LAB)
                  lab[...,0] = clahe.apply(lab[...,0])
                  pic=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            gaussian = np.random.normal(self.mean, (randomNumber)**0.5, self.dim)
            X = np.clip (X + gaussian, 0, 255).astype(np.uint8)
            for i in range (0, int(self.batch_size/2)):
                X[i] = rotate (X[i], ((randomNumber+i) % 13-7)*4, preserve_range=True).astype (np.uint8)
                y[i] = rotate (y[i], ((randomNumber+i) % 13-7)*4, preserve_range=True).astype (floatType)
        return X, y

train_generator = DataGenerator(x_train, y_train.astype(floatType), batch_size = 18, dim = input_shape, n_classes=2, shuffle=True)
val_generator = DataGenerator(x_val, y_val.astype(floatType), batch_size= 1, dim = input_shape,n_classes=2, shuffle=False)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes = 1, encoder_freeze=encoder_freeze, activation='sigmoid')

# load old weights
if (Load_weight == 1):
  model.load_weights('Best Model.hdf5')
  lr = 0.00002
else:
  if (encoder_freeze == True):
    lr = 0.01
  else:
    lr = 0.001

if (halfPrecision == 1):
  lr = 0.01
model_loss = sm.losses.bce_jaccard_loss
model_loss.smooth = epsilon
model.compile(
    #'adam',
    tf.keras.optimizers.Adam(lr,0.99,0.9999, epsilon=epsilon),
    #tf.keras.optimizers.Adam(0.0001,0.99,0.9999, epsilon=epsilon),
    #tf.keras.optimizers.SGD(learning_rate=0.0069, momentum=0.69, nesterov=False, name='SGD'),
    loss=model_loss,
    metrics=[sm.metrics.iou_score]
)

if (Load_opt == 1):
  model_train_vars = model.trainable_variables
  zero_grads = [tf.zeros_like(w) for w in model_train_vars]
  saved_vars = [tf.identity(w) for w in model_train_vars]
  model.optimizer.apply_gradients(zip(zero_grads, model_train_vars))
  [x.assign(y) for x,y in zip(model_train_vars, saved_vars)]
  opt = np.load(optimizer_file_name, allow_pickle=True)
  model.optimizer.set_weights(opt)
  model.load_weights('Best Model.hdf5')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="Best Model.hdf5",
    save_weights_only=False,
    monitor='val_iou_score',
    mode='max',
    save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/TfBoard', histogram_freq=1)
model.fit(
  train_generator,
  steps_per_epoch=len(train_generator),
  epochs=epochs,
  validation_data=val_generator,
  validation_steps=len(val_generator),
  callbacks=[model_checkpoint_callback, tensorboard_callback],
    verbose = 2)
print ("Yes")

import shutil
src="Best Model.hdf5"
dst="Best Model_bak.hdf5"
shutil.copy(src,dst)
np.save(optimizer_file_name, (model.optimizer.get_weights()))
