#!/usr/bin/env python
# Make sure that caffe is on the python path:
# For me: export PYTHONPATH=$PYTHONPATH:/home/nejat/PdC/2/caffe-master/python/

# source: https://groups.google.com/forum/#!searchin/caffe-users/python$2B$20classify$20an$20image$2B/caffe-users/Cl-LIlcSko4/vHF0MFxl5GEJ
# source: https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb

import caffe
import numpy as np
from skimage.transform import pyramid_gaussian
import skimage
import Image
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import DBSCAN
def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def getImagePyramid(image,downScale,window_length):
    # Nejat changed pyramid method because he needs an image instance in processImage() method
    # for croppping instead of ndarray computed by Quentin's method (pyramid_gaussian)
    w, h = image.size
    # maxLayer = min(math.floor(math.log(rows//36)//math.log(downScale)),\
    #   math.floor(math.log(cols//36)//math.log(downScale)))
    # return tuple(pyramid_gaussian(image,max_layer=maxLayer, downscale=downScale))
    imageList = [image]
    print "For original image:  => size: "+str(image.size)
    counter=1
    while True:
        w = int(w * downScale)
        h = int(h * downScale)
        if w>window_length and h>window_length:
            #Image.BICUBIC is one of the methods for resizing => I think it is best
            im = image.resize((w, h), Image.BICUBIC)
            #we can save this intermediate image => im.save("filename.jpeg")
            print "For intermediate image: "+str(counter)+" => size: "+str(im.size)
            imageList.append(im)
        else:
            break
        counter+=1
    return imageList
#end of the method

def findCentrumPoint(left, top, window_length):
    x = left+(window_length/2)
    y = top+(window_length/2)
    return (x, y)
#end of the method

def processImage(imageList,downScale,network,threshold,window_length,step):
    ### INPUT ###
    # ImageList: An image pyramid of given image
    # downScale: subsampling degree => we reduce the image size according to downScale
    # network: Caffe Convolutional Neurol Network Pretrained Model
    # threshold: with this threshold, we accept if there is face in window zone
    # window_length: the size of slicing window => square window
    # step: we slice our window with this step => step=1 means we look at each pixel

    ### OUTPUT ###
    # facesPosition: sorted and unique position list

    #TODO Save face size

    left_corner=0
    top_corner=0
    facesPosition = []
    scale_degree= 0  # truc = 0

    #a log file is good to follow each step
    with open('traces.log', 'w') as f:
        for image in imageList:
            width, height = image.size
            f.write("Image : " + str(scale_degree)+"\n")
            for left in xrange(0, width-window_length+1, step):
                for top in xrange(0, height-window_length+1, step):
                    cropped = image.crop((left, top, left+window_length,\
                            top+window_length))
                    cropped.load()
                    # we need to swap image dimensions for image to be loaded => transpose()
                    # because the dimensions of the image to be loaded is  (36, 36, 1).
                    # That is why we will use (2,1,0) order
                    # w.r.t our network expect theses dimensions: (1, 1, 36, 36)
                    # we can verify it by typing: net.blobs['data'].data.shape or facenet_deploy.prototxt
                    data = np.asarray(cropped)
                    img = data[:, :, np.newaxis]
                    img = skimage.img_as_float(img).astype(np.float32)
                    img = img.transpose((2,0,1))
                    #cropped = image[:,left:left+window_length,top:top +window_length]
                    network.blobs['data'].data[...] = img
                    out = network.forward()
                    # out['prob'][0] gives us the score tuple of non-face class 0 and face class 1
                    #so we are interested in out['prob'][0][1] for the score of class 1
                    f.write("left: "+str(left)+" top: "+str(top)+" => score: "+\
                            str(out['prob'][0][1])+" => image size: ["\
                            +str(image.size)+"]\n")

                    if out['prob'][0][1]>threshold:
                        x, y = findCentrumPoint(left, top, window_length)
                        scaled_centrum_x = int(x*(1/float(downScale) ** scale_degree)) # truc
                        scaled_centrum_y = int(y*(1/float(downScale) ** scale_degree)) # truc
                        f.write("Good! => centrum_x: "+str(x)+" centrum_y: "\
                                +str(y)+" new_x: "+str(scaled_centrum_x)+" new_y: "\
                                +str(scaled_centrum_y)+"\n")
                        # uniqueness of position list
                        if (scaled_centrum_x, scaled_centrum_y) not in facesPosition:
                            facesPosition.append([scaled_centrum_x, scaled_centrum_y])
                        #example: 50/0.8=40 => 40*(1/0.8)=50
            scale_degree += 1 # truc += 1

    return sorted(facesPosition) # sorted by x
#end of the method



##############
#### MAIN ####
##############

if __name__ == '__main__':

    ##################################
    #### PARAMETER CONFIGURATIONS ####
    ##################################

    downScale = 0.8
    threshold = 0.975 #for accepting if an image is a face-image or not
    window_length = 36
    step = 1 # for slicing window
    image_dir = "../multiple_faces/"
    filename = "Japan.jpg" # another example: "ucsd.png"
    filepath = image_dir + filename
    caffemodel = "../facenet_iter_505000.caffemodel"
    model = "../facenet_deploy.prototxt"

    ###############
    #### CAFFE ####
    ###############

    #set cpu mod
    caffe.set_mode_cpu()
    #load model (deploy) and weights (caffemodel)
    net = caffe.Net(model, caffemodel, caffe.TEST)

    #########################################
    #### LOADING IMAGE AND PREPROCESSING ####
    #########################################

    image = Image.open(filepath)
    #image = Image.open("../ucsd.png")
    data = np.asarray(image) # image in a matrix format

    #get image pyramid
    imageList = getImagePyramid(image,downScale,window_length)
    print "There are "+str(len(imageList))+" images in the pyramid (with original image)"


    ####################
    #### PROCESSING ####
    ####################

    positions = processImage(imageList,downScale,net,threshold,window_length,step)

    #print  positions
    print "In total, there are "+str(len(positions))+" unique positions"

    ########################
    #### POSTPROCESSING ####
    ########################

    ### STEP 1 ###

    # TODO clustering algorithm for finding the centroid of each dense group in positions
    # We will plot this new centroids in plot actually
    # We may do that by an 'Connected Component' algorithm
    # In that algorithm, the decision of creating a new group may be done with euclidian distance or weights
    #db =  DBSCAN(eps=3, min_samples=15).fit(positions)
    db =  DBSCAN(eps=3, min_samples=15).fit(positions)# why min_samples =10? because we take 'step' = 3 not 1

    clusters = []
    implot = plt.imshow(data,cmap = cm.Greys_r)
    for k in np.unique(db.labels_):
        clusters.append([])
        members = np.where(db.labels_ == k)[0]
        print members
        if k == -1:
            print("outliers:")
        else:
            #print("cluster %d:" % k)
            # we find the centrum point with reduce() for x and y separetly
            clusters[k] = [((reduce(lambda x,y: x+y,[positions[i][0] for i in members])/len(members)),\
                    (reduce(lambda x,y: x+y,[positions[i][1] for i in members])/len(members)))]

    colors = get_cmap(len(clusters))
    for k in xrange(len(clusters)):
            plt.scatter([pos[0] for pos in clusters[k]],[pos[1] for pos in clusters[k]],c=colors(k))
    plt.show()


#    for position in positions:
#        db = DBSCAN(eps=36, min_samples=10).fit(position)
#	labels = db.labels_
#        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
#        print len(postion)
#        print len(labels)

    ### STEP 2 ###

    # generating local pyramid based on each centroid found in clustring algorithm
    # and execute pretrained model again in the local pyramid (for each sub-image)
    # gathering scores of each local pyramid
    # deciding if there is a face or not in each local

    ##############
    #### PLOT ####
    ##############

    # Scatter plot for showing each postion on the given image

#    implot = plt.imshow(data,cmap = cm.Greys_r)
#    #colors = get_cmap(len(imageList))
#    #for pos in positions:
#        #plt.scatter(pos[0],pos[1],c=colors(pos[2]),)
#    for position in positions:
#        plt.scatter([pos[0] for pos in position],[pos[1] for pos in position])
#    plt.show()
