import i2v
from PIL import Image
import glob
import re
import pickle
import cv2
import numpy as np
import signal
import sys
from multiprocessing.dummy import Pool as ThreadPool

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

files = glob.glob('../../../dataset/data/*.png', recursive=True)
d = {}

# with open('features.pickle', 'rb') as handle:
#     d = pickle.load(handle)
# count = 0

l = len(files)
tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair',
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair',
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def thread_function(file):
    n = re.findall(r'\d+', file)[-1]
    if n in d:
        return
    
    global tag_dict

    img = Image.open(file)    
    img = np.array(img)

    # The images are of different size, the traid off is to rescale all the images at 128x128
    img2 = cv2.resize(img, (128, 128))

    feature = illust2vec.estimate_plausible_tags([img2], threshold=0.25)
    f = feature[0]['general']

    solo = False
    hair = False
    eye = False
    dictionary = {}
    for first, second in f:
        if first == 'solo':
            solo = True
        if first in tag_dict:
            dictionary[first] = 1
            if 'eyes' in first:
                eye = True
            if 'hair' in first:
                hair = True

    # if the image is not solo, or the estimator hasn't found the color of eye or hair, skip the image because we need both
    if solo == False or eye == False or hair == False:
        return

    d[str(n)] = dictionary
    return

# # When you do CTRL+C the program save the file, to start from where you stop it
# def signal_handler(sig, frame):
#     with open('features.pickle', 'wb') as handle:
#         pickle.dump(d, handle)
#     sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)


pool = ThreadPool(16) 
results = pool.map(thread_function, files)

pool.close() 
pool.join() 

with open('../features.pickle', 'wb') as handle:
    pickle.dump(d, handle)

print("Done")
