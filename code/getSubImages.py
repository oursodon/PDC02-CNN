#!/usr/bin/env python

from os import listdir
from os.path import isfile, join
import Image

path="Public/"
window=36

files = [f for f in listdir(path) if isfile(join(path, f))]
for f in files:
    im = Image.open(join(path, f))
    x, y = im.size
    for left in xrange(x/window):
        for top in xrange(y/window):
            box=(left*window,top*window,left*window+window,top*window+window)
            cropped=im.crop(box)
            fname=f.split(".")[0]
            fext=f.split(".")[1]
            cropped.save(join(path, fname+"_"+str(left)+"_"+str(top)+"."+fext))


