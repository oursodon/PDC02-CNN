#!/usr/bin/env python
# Make sure that caffe is on the python path:
# For me: export PYTHONPATH=$PYTHONPATH:/home/nejat/PdC/2/caffe-master/python/

# source: https://groups.google.com/forum/#!searchin/caffe-users/python$2B$20classify$20an$20image$2B/caffe-users/Cl-LIlcSko4/vHF0MFxl5GEJ
# source: https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb

import caffe
import numpy as np
from skimage.transform import pyramid_gaussian
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def getImagePyramid(image,downScale):
    dim, cols, rows = image.shape
    maxLayer = min(math.floor(math.log(rows//36)//math.log(downScale)),math.floor(math.log(cols//36)//math.log(downScale)))
    return tuple(pyramid_gaussian(image,max_layer=maxLayer, downscale=downScale))

def setBoxCorners(left_corner, top_corner, window_length):
    box=(left_corner,top_corner,left_corner+window_length,\
            top_corner+window_length)
    return box

def findCentrumPoint(left, top, window_length):
    x = (left+window_length)/2
    y = (top+window_length)/2
    return (x, y)

def fusionMatrixInOriginalScale(window_length, matrix_list):
    matrix_final = np.zeros((window_length, window_length))
    for matrix in matrix_list:
        print "TODO"
        # EGER MATRIX BOYUTU 36dan buyukse 36 cekmek lazim
    return matrix_final

def processImage(imageList,scale,network):
#TODO Save face size
    window_length=36 # original scale => (36, 36)
    left_corner=0
    top_corner=0

    facesPosition = []
    truc = 0

    for image in imageList:
        dim, width, height = image.shape
      
        for left in xrange(0,width-window_length+1,10):

            for top in xrange(0,height-window_length+1,10):
                #cropped = image.crop(setBoxCorners(left, top, window_length))
                cropped = image[:,left:left+window_length,top:top +window_length]
                #print cropped.shape
                network.blobs['data'].data[...] = cropped
                out = network.forward()
                result = out['prob'][0].argmax() # it is a face image #TODO change the line
                if result == 1:
                    #print "left:", left, "top:", top
                    x, y = findCentrumPoint(left, top, window_length)
                    #print x, y
                    facesPosition.append ( (x * (scale** truc) ,y *( scale ** truc ) ))

        truc += 1
        print "Image : " + str(truc)

    return facesPosition


caffe.set_mode_cpu()
net = caffe.Net("../facenet_deploy.prototxt", "../facenet_iter_200000.caffemodel", caffe.TEST)

# we need to swap image dimensions for image to be loaded => transpose()
# because the dimensions of the image to be loaded is  (36, 36, 1). That is why we will use (2,1,0) order
# w.r.t our network expect theses dimensions: (1, 1, 36, 36)
# we can verify it by typing: net.blobs['data'].data.shape or facenet_deploy.prototxt
image =  caffe.io.load_image("../imgTest.jpg", False).transpose((2,1,0))

imageList = getImagePyramid(image,1.2)

print image.shape
print image
print len(imageList)

positions = processImage(imageList,1.2,net)
print positions
implot = plt.imshow(image[0],cmap = cm.Greys_r)
plt.scatter([pos[0] for pos in positions],[pos[1] for pos in positions])
plt.show()
"""
net.blobs['data'].data[...] = image
out = net.forward()
print("For the image: 0/image01_36x36.pgm")
print("Predicted class is #{}.".format(out['prob'][0].argmax()))
#output is 0. So it is not a face image!

image2 = caffe.io.load_image("../1/image11_36x36.pgm" , False).transpose((2,1,0))
net.blobs['data'].data[...] = image2
out = net.forward()
print("For the image: 1/image11_36x36.pgm")
print("Predicted class is #{}.".format(out['prob'][0].argmax()))
#output is 1. So it is a face image!
"""
