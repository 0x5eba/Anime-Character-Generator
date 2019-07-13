import csv
import pickle
import glob, re
import numpy as np
import pandas as pd

tags = ['num', 'orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair',
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair',
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

imgs = glob.glob('../../dataset/data/*.png')
imgs_nums = set(re.findall(r'\d+', img)[-1] for img in imgs)

with open('features.pickle', 'rb') as handle:
    d = pickle.load(handle)

with open('features.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=tags)
    writer.writeheader()

    for k, v in d.items():
        if not k in imgs_nums:
            continue
            
        label = {k2: 0 for k2 in tags}
        label['num'] = k
        for name_feature, zero_or_one in v.items():
            # zero_or_one is always 1
            label[name_feature] = 1
        writer.writerow(label)