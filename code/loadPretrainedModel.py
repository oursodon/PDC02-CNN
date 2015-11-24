#!/usr/bin/env python
# Make sure that caffe is on the python path:
# For me: export PYTHONPATH=$PYTHONPATH:/home/nejat/PdC/2/caffe-master/python/

# source: https://groups.google.com/forum/#!searchin/caffe-users/python$2B$20classify$20an$20image$2B/caffe-users/Cl-LIlcSko4/vHF0MFxl5GEJ
# source: https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb

import caffe
import numpy as np

caffe.set_mode_cpu()
net = caffe.Net("../facenet_deploy.prototxt", "../facenet_iter_200000.caffemodel", caffe.TEST)

# we need to swap image dimensions for image to be loaded => transpose()
# because the dimensions of the image to be loaded is  (36, 36, 1). That is why we will use (2,1,0) order
# w.r.t our network expect theses dimensions: (1, 1, 36, 36)
# we can verify it by typing: net.blobs['data'].data.shape or facenet_deploy.prototxt
image =  caffe.io.load_image("../0/image01_36x36.pgm", False).transpose((2,1,0))
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
