from DatasetPrepare import *
import pandas as pd

def overlay (image, edge):
    for a in range (0, image.shape [0]):
        for b in range (0, image.shape [1]):
            if (edge [a, b] > 0):
                image[a, b, 1]=edge[a, b]
    image = np.clip (image, 0, 255)
    return image

def edgeDetector (image):
    gay = image
    gay = cv2.blur(image,(3, 3))
    #gay = cv2.blur(gay,(3, 3))
    edges = cv2.Canny(gay, 0,128, apertureSize = 3)
    gay = overlay (gay, edges)
    return gay, edges

def testEdgeDetector ():
    imageList = readAllImage("L:\\JAV Folder\\testFolderDeleteThis")
    for image in imageList:
        gay = edgeDetector (image)
        cv2.imshow ("Input", gay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def CleanPrediction (result, edge, radius = 15, threshold = 5):
    if (radius < 5):
        print ("Radius must be bigger than 4")
        radius = 5
    predict = np.copy (result)
    temp_a = [-1, -1, -1, 0, 0, 1, 1, 1]
    temp_b = [-1, 0, 1, -1, 1, -1, 0, 1]
    for ix in range (radius, edge.shape[0] - radius, 3):
        for iy in range (radius, edge.shape [1] - radius, 3):
            if (edge [ix, iy] > 0):
                #Kiem tra neu phan tu mang Edge nam hoan toan ben trong heatmap cua prediction
                check = False
                for fuckme in range (1, int (radius / 5 + 1)):
                    for itemp in range (0, len(temp_a)):
                        if (predict[ix + temp_a [itemp] * fuckme, iy + temp_b [itemp] * fuckme] == 0):
                            check = True
                            break
                    if (check == True):
                        break
                if (check == False):
                    continue
                
                if (predict [ix - radius, iy] > threshold):
                    for itemp in range (1, radius):
                        result [ix - itemp, iy] = 255
                        result [ix - itemp, iy-1] = 255
                        result [ix - itemp, iy+1] = 255
                if (predict [ix - int(radius/2), iy] > threshold):
                    for itemp in range (1, int(radius/2)):
                        result [ix - itemp, iy] = 255
                        result [ix - itemp, iy-1] = 255
                        result [ix - itemp, iy+1] = 255
                        
                if (predict [ix, iy - radius] > threshold):
                    for itemp in range (1, radius):
                        result [ix, iy - itemp] = 255
                        result [ix+1, iy - itemp] = 255
                        result [ix-1, iy - itemp] = 255
                if (predict [ix, iy - int(radius/2)] > threshold):
                    for itemp in range (1, int(radius/2)):
                        result [ix, iy - itemp] = 255
                        result [ix+1, iy - itemp] = 255
                        result [ix-1, iy - itemp] = 255
                        
                if (predict [ix + radius, iy] > threshold):
                    for itemp in range (1, radius):
                        result [ix + itemp, iy] = 255
                        result [ix + itemp, iy-1] = 255
                        result [ix + itemp, iy+1] = 255
                if (predict [ix + int(radius/2), iy] > threshold):
                    for itemp in range (1, int(radius/2)):
                        result [ix + itemp, iy] = 255
                        result [ix + itemp, iy-1] = 255
                        result [ix + itemp, iy+1] = 255
                        
                if (predict [ix, iy + radius] > threshold):
                    for itemp in range (1, radius):
                        result [ix, iy + itemp] = 255
                        result [ix+1, iy + itemp] = 255
                        result [ix-1, iy + itemp] = 255
                if (predict [ix, iy + int(radius/2)] > threshold):
                    for itemp in range (1, int(radius/2)):
                        result [ix, iy + itemp] = 255
                        result [ix+1, iy + itemp] = 255
                        result [ix-1, iy + itemp] = 255
    return result
                    
                    
            
#testEdgeDetector()
