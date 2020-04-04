from __future__ import print_function, division
import os
import math
import torch
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


def clean_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    return dir


def clean_file(fname):
    if os.path.isfile(fname):
        os.remove(fname)
    return fname


def savecsv(info, filepath):
    df = pd.DataFrame(info)
    df.to_csv(clean_file(filepath), sep=' ', index=None, header=['Filename', 'ML_Both_Arms', 'HL_Activity'])
    print("save file", filepath)


data_path = '/home/grads/a/a.pakbin/DATA/OPPORTUNITY'
data_files = [f for f in os.listdir(data_path) if f.endswith('.dat')]

csv_info = []
save_path = clean_dir('/home/grads/a/a.pakbin/DATA/OPPORTUNITY/MIX/preprocessed')


def most_common(lst):
    return max(set(lst), key=lst.count)


ML_dict = {
    0: 0,  # null class
    406516: 1,  # Open Door 1
    406517: 2,  # Open Door 2
    404516: 3,  # Close Door 1
    404517: 4,  # Close Door 2
    406520: 5,  # Open Fridge
    404520: 6,  # Close Fridge
    406505: 7,  # Open Dishwasher
    404505: 8,  # Close Dishwasher
    406519: 9,  # Open Drawer 1
    404519: 10,  # Close Drawer 1
    406511: 11,  # Open Drawer 2
    404511: 12,  # Close Drawer 2
    406508: 13,  # Open Drawer 3
    404508: 14,  # Close Drawer 3
    408512: 15,  # Clean Table
    407521: 16,  # Drink from Cup
    405506: 17,  # Toggle Switch
}

HL_dict = {
    0: 0,  # null class
    101: 1,  # Relaxing
    102: 2,  # Coffee time
    103: 3,  # Early morning
    104: 4,  # Clean up
    105: 5,  # Sandwich time
}

for fname in data_files:
    # content = open(os.path.join(data_path, fname)).readlines()
    content = np.loadtxt(os.path.join(data_path, fname), delimiter=' ')
    imp = Imputer()
    content = imp.fit_transform(content)

    for n in tqdm(range(int((len(content) - 30) / 3))):
        newfile = "%s_%s.%s" % (fname.split('.')[0], n, 'dat')

        labels = [[d[-1], d[-6]] for d in content[3 * n:3 * n + 30]]
        labels = [[ML_dict[int(label[0])], HL_dict[int(label[1])]] for label in labels]
        labels = [most_common([int(d[i]) for d in labels]) for i in range(len(labels[0]))]
        # now labels [0] is the label of mid-level activity
        #     labels [1] is the label of high-level activity

        # we need the activity and context (high-level activity) exist at the
        # same time
        if labels[0] == 0 or labels[1] == 0:
            continue

        # data = [[math.inf if s == 'NaN' else float(s) for s in d.strip().split(' ')[1:134]] for d in
        #         content[3 * n:3 * n + 30]]
        data = [d[1:134] for d in content[3 * n: 3 * n + 30]]

        np.save(os.path.join(save_path, newfile), np.array(data))

        csv_info.append([newfile] + labels)

savecsv(csv_info, os.path.join(save_path, "info.csv"))

info_train, info_test = train_test_split(csv_info, test_size=0.2, random_state=42)
savecsv(info_train, os.path.join(save_path, "info_train.csv"))
savecsv(info_test, os.path.join(save_path, "info_test.csv"))
