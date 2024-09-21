import os, time
print(os.getcwd())
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
#import keras
import tensorflow as tf
import DatasetPrepare
from tensorflow import keras
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
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
BACKBONE = 'efficientnetb0'
preprocess_input = sm.get_preprocessing(BACKBONE)
input_shape = (256, 256, 3)

Load_opt = 1
Load_weight = 1
encoder_freeze = False
epochs = 200
batch_size = 6
optimizer_file_name = 'optimizer.npy'
# load your data
#DatasetPrepare || LoadDataset
x, y = DatasetPrepare.LoadDataset("L:\\JAV Folder\\Test Frames")
y = y.astype(np.float32)
#x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=69)

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
        return X, y

val_generator = DataGenerator(x, y.astype(floatType), batch_size= 1, dim = input_shape,n_classes=2, shuffle=False)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes = 1, encoder_freeze=encoder_freeze, activation='sigmoid')

# load old weights
if (Load_weight == 1):
  model.load_weights('Best Model.hdf5')
  lr = 0.0001
else:
  if (encoder_freeze == True):
    lr = 0.01
  else:
    lr = 0.001

if (halfPrecision == 1):
  lr = 0.1
model_loss = sm.losses.bce_jaccard_loss
model_loss.smooth = epsilon
model.compile(
    #'adam',
    tf.keras.optimizers.Adam(lr,0.99,0.9999, epsilon=epsilon),
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
start = time.time()
gae = model.predict(
  val_generator,
    verbose = 2)
Big_ImageList, Big_JsonList = DatasetPrepare.LoadDatasetCustom(
                                      "L:\JAV Folder\Test Frames",
                                      ImageName = "ImageList_512_Binary.npy",
                                      JsonName = "JsonList_512_Binary.npy")
predictList = np.zeros ((gae.shape[0], 512, 512))
for i in range (0, gae.shape[0]):
  predictList [i] = cv2.resize (gae[i], (512, 512))
np.expand_dims (predictList, -1)
predictList = np.expand_dims ((predictList*255).astype(int), -1)
Big_ImageList = np.concatenate ((Big_ImageList, predictList), 3)
np.save ("L:\\jav folder\\Test Frames\\Combined", Big_ImageList)
print(f'Time taken to train: {time.time() - start}')
print (time.time() - start)
print (gae.shape)
#print ("Yes")

import shutil
src="Best Model.hdf5"
dst="Best Model_bak.hdf5"
shutil.copy(src,dst)
np.save(optimizer_file_name, (model.optimizer.get_weights()))
