'''
This code is for the complete tree generation version
'''

from argparse import ArgumentParser
import numpy as np
import glob
import os
import cvxpy as cp
import pickle
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.morphology import remove_small_objects
from scipy.sparse import csr_matrix, coo_matrix, vstack
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
from datetime import datetime

from scipy.ndimage import grey_dilation


def EMD_match(img1, img2):

    img1_idx = np.unique(img1)
    img1_idx = img1_idx[img1_idx>0]
    list1, list1_name = [], []

    for node in img1_idx:
        cell = np.where(img1 == node)
        minx, maxx, miny, maxy = min(cell[0]), max(cell[0]), min(cell[1]), max(cell[1])
        if maxx > minx and maxy > miny:
            list1_name.append((0, node))
            list1.append(tuple(cell))

    img2_idx = np.unique(img2)
    img2_idx = img2_idx[img2_idx > 0]
    list2, list2_name = [], []

    for node in img2_idx:
        cell = np.where(img2 == node)

        minx, maxx, miny, maxy = min(cell[0]), max(cell[0]), min(cell[1]), max(cell[1])
        if maxx > minx and maxy > miny:
            list2.append(tuple(cell))
            list2_name.append((0, node))


    weight = []
    recorder = [[], []]

    ### find possible matching pairs
    for k, node2 in enumerate(list2):
        x2, y2 = np.mean(node2[0]), np.mean(node2[1])
        set2 = set(np.ravel_multi_index(node2, img1.shape))
        for j, node1 in enumerate(list1):
            # x1, y1 = np.mean(node1[0]), np.mean(node1[1])
            set1 = set(np.ravel_multi_index(node1, img1.shape))
            IoU = float(len(set2.intersection(set1))) / len(set2.union(set1))
            diff = abs(len(set2) - len(set1)) / len(set1)
            if IoU > 0.2 and diff < 0.5:
                # temp = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
                sim = IoU
                weight.append(sim)
                recorder[0].append(j)
                recorder[1].append(k)

    if len(weight) > 0:
        Branch = len(list1) + len(list2)
        print('length of weight: ', len(weight))
        weight = np.array(weight)
        recorder = np.array(recorder)

        Aeq = np.zeros((Branch, len(weight)))
        Beq = np.ones(Branch)
        for j in range(len(weight)):
            sIdx = recorder[0][j]
            tIdx = recorder[1][j] + len(list1)
            ### branches in source
            Aeq[sIdx][j] = 1
            Aeq[tIdx][j] = 1


        X = cp.Variable(len(weight), boolean = True)
        constraints = [Aeq@X<=Beq]

        prob = cp.Problem(cp.Maximize(weight.T@X), constraints)
        prob.solve('GLPK_MI')

        for idx in range(len(X.value)):
            if X.value[idx] > 0:
                img2[list2[recorder[1][idx]]] = 0

    return img2



