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

def LoadDatasetCustom (path_to_dataset, ImageName = "ImageList_512_Binary.npy", JsonName = "JsonList_512_Binary.npy"):
    print ("Reading preloaded BINARY dataset")
    ImageList = np.load(path_to_dataset + "\\" + ImageName)
    JsonList = np.load(path_to_dataset + "\\" + JsonName)
    return ImageList, JsonList

def readAllImage (path_to_image, height=540, width=960):
    count = 0
    for Files in glob.glob(path_to_dataset + "\\*\\*.png"):
            count+=1
    ImageList = numpy.empty ((count, height, width, 3), dtype=np.uint8)
    count = 0
    for Files in glob.glob(path_to_dataset + "\\*\\*.png"):
            Image = cv2.imread(Files)
            #ImageList[count] = np.expand_dims(cv2.resize(Image, (height, width)),0)
            ImageList[count] = np.expand_dims(cv2.resize(Image, (width, height)),0)
            count+=1
    return ImageList

def DatasetPrepare (path_to_dataset, height, width, saveToFile = 1, is_binary = 0,Postfix = ""):
    count = 0
    bak_count = 0
    for Folder in glob.glob(path_to_dataset + "\\*\\"):
        bak_count = count
        for JsonFiles in glob.glob (Folder + "Result\\*"):
            count+=1
        #print (-bak_count + count, " " ,Folder)
        #print("")
    print ("Counted [DatasetPrepare]")
    print (count)
    #return 0,0
    if (is_binary == 0):
        ImageList = numpy.empty ((count, height, width, 3), dtype=np.uint8)
    else:
        ImageList = numpy.empty ((count, height, width, 1), dtype=np.uint8)
    JsonList = numpy.empty ((count, height, width, 1), dtype=np.bool_)
    count = 0
    for Folder in glob.glob(path_to_dataset + "\\*\\"):
        for JsonFiles in glob.glob (Folder + "Result\\*"):
            try:
                JsonData = ReadJson (JsonFiles)
            except:
                print ("Error")
                print (JsonFiles)
                print (Folder)
                continue
            if (is_binary == 0):
                Image = cv2.resize(
                            cv2.imread(Folder + ((JsonFiles.split("\\"))[-1])[0:-5] + ".png")
                                   ,(height, width))
            else:
                Image =np.expand_dims(
                        cv2.resize(
                            cv2.imread(Folder + ((JsonFiles.split("\\"))[-1])[0:-5] + ".png",
                                        cv2.IMREAD_GRAYSCALE)
                        ,(height, width))
                       ,-1)
                
            ImageList[count] = np.expand_dims(Image,0)
            JsonList[count] = np.expand_dims(skimage.transform.resize(JsonData, (height, width),anti_aliasing= False),0)
            count+=1
    if (saveToFile == 1):
        np.save(path_to_dataset + "\\ImageList" + Postfix, ImageList)
        np.save(path_to_dataset + "\\JsonList" + Postfix, JsonList)
    return ImageList, JsonList
if __name__ == "__main__":
    #print ("Running")
    #start = time.time()
    ImageList, JsonList = DatasetPrepare (path_to_dataset, 512, 512,
                                          saveToFile = 1, is_binary = 1,Postfix = "_512_Binary")

    #print (ImageList.dtype)
    #print (JsonList.dtype)
    #gay1, gay2 = LoadDataset (path_to_dataset)
    #print (gay1.dtype)
    #print (gay2.dtype)

    #cv2.imshow ("JsonData", ImageList[count].astype(np.uint8))
    #cv2.imshow ("JsonData", JsonList[count].astype(np.uint8)*255)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print ("Done")
    #print(f'Time Taken: {time.time() - start}')
    #print (ImageList.shape)
    #print (JsonList.shape)

    #mask = ReadJson (path_to_json)
    #_,ret = cv2.imencode('.jpg', mask[:,:,0]*255)

    #i = IPython.display.Image(data=ret)
    #IPython.display.display(i)
