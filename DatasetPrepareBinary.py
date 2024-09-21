# TEST READ JSON


import os
import sys
import random
import math
import re
import json
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import glob
import skimage.color
import skimage.io
import skimage.transform
import IPython
from joblib import Parallel, delayed

path_to_json = "L:\\JAV Folder\\Test Frames\\CAWD-152\\Result\\frame145650.json"
path_to_dataset = "L:\JAV Folder\Test Frames"
def ReadJson (path_to_json):
    annotation = json.load(open(path_to_json))
    mask = np.zeros([annotation["imageHeight"], annotation["imageWidth"], 1], dtype=np.bool_)

    for i in range(len (annotation["shapes"])):
        if ((annotation["shapes"][i]["label"] == 'A') or (annotation["shapes"][i]["label"] == "Human")):
            pointsx,pointsy=zip(*annotation["shapes"][i]["points"])
            rr, cc = skimage.draw.polygon(pointsx, pointsy, shape = (1920, 1080))
            mask[cc, rr, 0] = 1
            
    for i in range(len (annotation["shapes"])):
        if ((annotation["shapes"][i]["label"] != 'A') and (annotation["shapes"][i]["label"] != "Human")):
            pointsx,pointsy=zip(*annotation["shapes"][i]["points"])
            rr, cc = skimage.draw.polygon(pointsx, pointsy, shape = (1920, 1080))
            mask[cc, rr, 0] = 0
    return mask

def LoadDataset (path_to_dataset):
    print ("Reading preloaded dataset")
    ImageList = np.load(path_to_dataset + "\\ImageList.npy")
    JsonList = np.load(path_to_dataset + "\\JsonList.npy")
    return ImageList, JsonList

def LoadDatasetBinary (path_to_dataset):
    print ("Reading preloaded BINARY dataset")
    ImageList = np.load(path_to_dataset + "\\ImageListBinary.npy")
    JsonList = np.load(path_to_dataset + "\\JsonList.npy")
    return ImageList, JsonList

def readOneImage (path_to_image):
    return cv2.resize(cv2.imread(path_to_image), (512, 512))
def BinaryImage (path_to_image):
    return cv2.resize(cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE), (512, 512))
def ClaheImage (path_to_image):
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(5,5))
    return cv2.resize(cv2.GaussianBlur(clahe.apply(cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)),(7,7),0), (512, 512))
def JsonImage (path_to_json):
    JsonData = ReadJson (path_to_json)
    return (skimage.transform.resize(JsonData, (512, 512),anti_aliasing= False))
    
    
def readAllImage (path_to_image, height=540, width=960):
    
    ImageList = []
    nprocs = 8
    gay = glob.glob(path_to_dataset + "\\*\\*.png")
    
    ImageList.extend(Parallel(n_jobs=nprocs)(delayed(ClaheImage)(gay[idx]) for idx in range(len(gay))))
    return ImageList

def DatasetPrepareBinary (path_to_dataset, height, width):
    ImageList = [] #numpy.empty ((count, height, width, 3), dtype=np.uint8)
    nprocs = 8  
    JsonPath = glob.glob(path_to_dataset+ "\\*\\Result\\*")
    ImagePath = [path_to_dataset + "\\" + (gay.split("\\"))[-3] + "\\" + ((gay.split("\\"))[-1])[0:-5] + ".png" for gay in JsonPath]
      
    ImageList.extend(Parallel(n_jobs=nprocs)(delayed(ClaheImage)(ImagePath[idx]) for idx in range(len(ImagePath))))
    ImageList = np.asarray(ImageList, dtype = np.uint8)

    ImageList = np.broadcast_to(ImageList[...,None], ImageList.shape+(3,))
    
    np.save(path_to_dataset + "\\ImageListBinary", ImageList)  
    return ImageList

def DatasetPrepare (path_to_dataset, height, width):    
    ImageList = [] #numpy.empty ((count, height, width, 3), dtype=np.uint8)
    nprocs = 8  
    JsonPath = glob.glob(path_to_dataset+ "\\*\\Result\\*")
    ImagePath = [path_to_dataset + "\\" + (gay.split("\\"))[-3] + "\\" + ((gay.split("\\"))[-1])[0:-5] + ".png" for gay in JsonPath]
    ImageList.extend(Parallel(n_jobs=nprocs)(delayed(readOneImage)(ImagePath[idx]) for idx in range(len(ImagePath))))
    ImageList = np.asarray(ImageList, dtype = np.uint8)
    np.save(path_to_dataset + "\\ImageList", ImageList)
    return ImageList
def JsonPrepare (path_to_dataset, height, width):
    JsonList = []
    nprocs = 8  
    JsonPath = glob.glob(path_to_dataset+ "\\*\\Result\\*")
    JsonList.extend(Parallel(n_jobs=nprocs)(delayed(JsonImage)(JsonPath[idx]) for idx in range(len(JsonPath))))
    JsonList = np.asarray(JsonList, dtype = np.bool_)
    np.save(path_to_dataset + "\\JsonList", JsonList)
    return JsonList
    
ImageList = DatasetPrepare ("L:\\JAV Folder\\Test Frames", 512, 512)
BinaryImageList = DatasetPrepareBinary("L:\\JAV Folder\\Test Frames", 512, 512)
JsonList = JsonPrepare("L:\\JAV Folder\\Test Frames", 512, 512)
"""
def equalizeImage(img):
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(5,5))
    return cv2.GaussianBlur(clahe.apply(img),(7,7),0)

x_rgb, y = LoadDataset("L:\\JAV Folder\\Test Frames")
x = (x_rgb[:,:,:,0] * 0.114 + x_rgb[:,:,:,1] * 0.587 + x_rgb[:,:,:,2] * 0.299).astype (np.uint8)
nprocs = 8
result = []
result.extend(Parallel(n_jobs=nprocs)(delayed(equalizeImage)(x[idx]) for idx in range(x.shape[0])))
np.save("L:\\JAV Folder\\Test Frames\\ImageListBinary", result)"""

