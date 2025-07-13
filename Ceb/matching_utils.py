import os
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.morphology import thin, remove_small_objects
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle, LevelOrderIter
from skimage.io import imread, imsave, use_plugin
from skimage.morphology import thin, dilation
from collections import defaultdict, OrderedDict

from EMD import EMD_match

from itertools import combinations

def color_img(img, color_platte):

    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    unique = np.unique(img)
    unique = unique[unique > 0]

    for idx in unique:
        for k in range(3):
            result[:,:,k][img == idx] = color_platte[idx][k]

    return result

def debug(pairs, img, dict2, ref_img_reorder):

    gt = np.zeros(img.shape, dtype = np.uint16)
    pred = np.zeros(img.shape, dtype=np.uint16)

    for time, k in enumerate(pairs):
        gt[ref_img_reorder==k[0]] = time + 1
        item = dict2[k[1]]
        for i in item:
            pred[img == i] = time + 1

    color_platte = np.random.randint(0, 255, (len(pairs) + 5, 3))
    gt = color_img(gt, color_platte)
    pred = color_img(pred, color_platte)
    imsave('gt.png', gt)
    imsave('pred.png', pred)

def GetComFast(img, pm, gt, graph):
    '''
    :param img:
    :param min_cc:
    :return: wsr: 1xC (C: # of watershed values)
             node_dict: NxM
    '''
    ### get watershed value list
    wsr = np.unique(img)
    wsr = wsr[wsr > 0]
    wsr.sort()

    if len(wsr) < 1:
        return [], []
    else:
        node_list, including_list = [], []
        ### get labeled images
        pm_index = label(pm > 127)
        indexs = np.unique(pm_index)
        indexs = indexs[indexs>0]

        for idx in indexs:
            pred_cover = np.unique(img[pm_index==idx])
            pred_cover = pred_cover[pred_cover > 0]

            gt_cover = np.unique(gt[pm_index==idx])
            gt_cover = gt_cover[gt_cover > 0]

            ## get combination only covered by gt to reduce search space
            for gt_i in gt_cover:
                cover = np.unique(img[gt == gt_i])
                cover = set(cover).intersection(set(pred_cover))

                for i in range(len(cover)):
                    possibles = combinations(cover, i+1)
                    # check the valid of each combination
                    for item in possibles:
                        stack = [item[0]]
                        left_node = set(t for t in item[1:])
                        while stack:
                            cur = stack.pop()
                            for child in graph[cur]:
                                if child in left_node:
                                    stack.append(child)
                                    left_node.remove(child)

                        if len(left_node) == 0:
                            node_list.append(item)


        for idx in wsr:
            temp = []
            for t, node in enumerate(node_list):
                if idx in node:
                    temp.append(t)
            including_list.append(temp)
        return including_list, node_list


def get_ref(img, base_img):

    img = EMD_match(base_img, img)
    index = np.unique(img)
    index = index[index > 0]

    list_all = []
    for node in index:
        cell = np.where(img == node)
        list_all.append(tuple(cell))

    treeIdx = [i for i in range(len(list_all))]

    return list_all, treeIdx

def get_ref_noEMD(img):

    ## use EMD to remove redundant matching
    index = np.unique(img)
    index = index[index > 0]

    list_all = []
    for node in index:
        cell = np.where(img == node)
        list_all.append(np.ravel_multi_index(cell, img.shape))

    return list_all

def get_neighbors(height, width, pixel):
      return np.mgrid[
         max(0, pixel[0] - 1):min(height, pixel[0] + 2),
         max(0, pixel[1] - 1):min(width, pixel[1] + 2)
      ].reshape(2, -1).T


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

