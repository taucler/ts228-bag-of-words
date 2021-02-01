import cv2
import numpy as np
import os
import pandas as pd
import csv

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

img_path = './home/maryem/s8/ts228-bag-of-words/entrainement/'
train = pd.read_csv('./home/maryem/s8/ts228-bag-of-words/entrainement/avions.csv')
species = train.species.sort_values().unique()

dico = []

def step1():
    for leaf in train.id:
        img = cv2.imread(img_path + str(leaf) + ".jpg")
        kp, des = sift.detectAndCompute(img, None)

        for d in des:
            dico.append(d)
