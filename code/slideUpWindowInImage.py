#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import Image
from os import listdir
from os.path import isfile, join

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

np.set_printoptions(threshold='nan') # to print all numpy array

window_length=34 # original scale => (36, 36)
left_corner=0
top_corner=0
#image_files= ["deneme/image31_36x36.pgm", "deneme/image31_43x43.pgm","deneme/image31_51x51.pgm"]
image_files= ["deneme/image31_43x43.pgm"]
matrix_list = []

for image_file in image_files:
    im = Image.open(image_file)
    width, height = im.size
    # create width x height matrice => to fill with 1 is the cropped image
    # is a face image
    m = np.zeros((width, height))
    for left in xrange(width-window_length+1):
        print "iter:"
        for top in xrange(height-window_length+1):
            cropped = im.crop(setBoxCorners(left, top, window_length))
            #TODO add classifier method
            result = 1 # it is a face image #TODO change the line
            if result == 1:
                print "left:", left, "top:", top
                x, y = findCentrumPoint(left, top, window_length)
                print x, y
                m[x, y] = 1 # something different than 0 to show a face image found
    matrix_list.append(m) # there should be (width-window_length)*(height-window_length) matrix

#for mat in matrix_list:
#    print mat
#fusionMatrixInOriginalScale(window_length, matrix_list) # TODO
