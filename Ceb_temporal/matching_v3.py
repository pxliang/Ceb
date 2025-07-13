"""
base_img directory:
--base
    --iter0
        --stack0
        --stack1
        ...
    --iter1
        --stack0
        --stack1
        ...
    ...
"""
import sys

import numpy as np
import glob
import os
import cvxpy as cp
import pickle
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.transform import resize
from scipy.sparse import csr_matrix, coo_matrix, vstack
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from argparse import ArgumentParser
import time
from datetime import datetime
from scipy.ndimage import grey_dilation
from skimage.morphology import thin, remove_small_objects
from scipy.io import loadmat

from itertools import permutations, combinations

from tem_utils import get_ref, GetComFast_valid, get_sub_region
from find_reliable import boundary_class_reliable, tree_reliable


def get_ins_to_line(value_to_line):
    ins_to_line, line_to_label = defaultdict(set), defaultdict(dict)

    for item in value_to_line:
        line = value_to_line[item]
        ins_to_line[line[0]].add(line)
        ins_to_line[line[1]].add(line)
        line_to_label[line]['value'] = item
        line_to_label[line]['label'] = -1


    return ins_to_line, line_to_label

def main(args):

    sub_dir = [f for f in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, f))]
    sub_dir.sort()
    if len(sub_dir) == 0:
        sub_dir = ['']

    if args.get_base == 1:
        os.makedirs(os.path.join(args.base_img, 'iter-1'), exist_ok=True)
        # tree_reliable(args.img_dir, os.path.join(args.base_img, 'iter-1'), args.min_thres, args.min_region)
        boundary_class_reliable(args.watershed_dir, args.watershed_line_dir, args.watershed_map_dir, args.reliable_file, \
                                 args.low_thres, args.high_thres, os.path.join(args.base_img, 'iter-1'))

    for iter in range(args.start, args.iter):
        for item in sub_dir:
            os.makedirs(os.path.join(args.base_img, 'iter' + str(iter), item), exist_ok=True)
            imgname = [f for f in os.listdir(os.path.join(args.img_dir, item)) if f.endswith('.mat') or f.endswith('.png')]
            imgname.sort()
            base = [f for f in os.listdir(os.path.join(args.base_img, 'iter'+str(iter-1), item)) if f.endswith('.png') or f.endswith('.tif')]
            base.sort()

            watershed_names = [f for f in os.listdir(os.path.join(args.watershed_dir, item)) if
                               f.endswith('.png') or f.endswith('.tif')]
            watershed_names.sort()

            if args.valid_stop == 0:
                valid_stop = len(imgname)
            else:
                valid_stop = args.valid_stop

            assert len(imgname) == len(base)
            for i in range(len(imgname)):
                img_name_left, img_name_right = None, None
                watershed_name_middle = os.path.join(args.watershed_dir, item, watershed_names[i])
                pm_name_middle = os.path.join(args.img_dir, item, imgname[i])
                watershed_line_name_middle = os.path.join(args.watershed_map_dir, item, base[i][:-4]+'.pickle')

                if i >= args.valid_start and i <= valid_stop:
                    if args.left_pro and i > 0:
                        img_name_left = os.path.join(args.base_img, 'iter'+str(iter-1), item, base[i-1])
                    if args.right_pro and i + 1 < len(imgname):
                         img_name_right = os.path.join(args.base_img, 'iter' + str(iter-1), item, base[i+1])

                base_img_name = os.path.join(args.base_img, 'iter' + str(iter - 1), item, base[i])
                result_name = os.path.join(args.base_img, 'iter' + str(iter), item, base[i][:-4]+'.tif')
                HEMD(img_name_left, img_name_right, watershed_name_middle, watershed_line_name_middle, pm_name_middle, base_img_name, result_name)


def HEMD(img_name_left, img_name_right, watershed_name_middle, watershed_line_name_middle, pm_name_middle, base_img_name, result_name):

    # current image
    watershed_img = imread(watershed_name_middle)

    ## lines: line_id: (ins1, ins2)
    with open(watershed_line_name_middle, 'rb') as handle:
        lines = pickle.load(handle)


    ## line_to_label: (ins1, ins2):{'value': line_id, 'label': -1}
    _, line_to_label = get_ins_to_line(lines)

    ## adjcent matric: region as node, line as edge
    graph = defaultdict(set)
    for item in lines:
        n1, n2 = lines[item]
        graph[n1].add(n2)
        graph[n2].add(n1)

    if pm_name_middle.endswith('.mat'):
        pm_img = loadmat(pm_name_middle)['prob']
        pm_img = np.uint8(pm_img * 255)
    else:
        pm_img = imread(pm_name_middle)

    # current reliable image
    base_img = imread(base_img_name)
    watershed_img[base_img > 0] = 0

    # watershed id to region id
    idx_to_region = get_sub_region(watershed_img, pm_img)

    # the previous reliable image
    if img_name_left is not None:
        img_left = imread(img_name_left)
        ## do relibale-reliable matching
        list_left = get_ref(img_left, base_img)
    else:
        list_left = None

    # the next reliable image
    if img_name_right is not None:
        img_right = imread(img_name_right)
        ## do relibale-reliable matching
        list_right = get_ref(img_right, base_img)
    else:
        list_right = None

    list_refs = [list_left, list_right]
    # for left and then for right

    sels, scores = [], []
    region_dict = defaultdict(dict)
    for f_c, list_gt in enumerate(list_refs):
        if list_gt is None:
            sels.append([])
            scores.append(0)
        else:
            ## wsr_all indicates each ws region belong to which combinations
            ## node_list_all contains all valid ws region combinations
            wsr_all, node_list_all = GetComFast_valid(watershed_img, list_gt, graph)

            list2, list2_idx = [], []
            for node in node_list_all:
                cell_temp = []
                for n in node:
                    cell = np.where(watershed_img == n)
                    minx, maxx, miny, maxy = min(cell[0]), max(cell[0]), min(cell[1]), max(cell[1])
                    if maxx > minx and maxy > miny:
                        cell_temp.extend(np.ravel_multi_index(cell, watershed_img.shape))
                list2.append(cell_temp)
                list2_idx.append(node)

            sel, score = ref_can_matching(list_gt, list2, list2_idx, wsr_all)
            for k, f in enumerate(sel):
                if f > 0:
                    selected = node_list_all[k]
                    region_idx = idx_to_region[selected[0]]

                    ## check all selected region belong to the same 0.5 connected component
                    for s in selected:
                        assert idx_to_region[s] == region_idx
                    if f_c not in region_dict[region_idx]:
                        region_dict[region_idx][f_c] = dict()
                        region_dict[region_idx][f_c]['weight'] = 0
                        region_dict[region_idx][f_c]['sel'] = []
                    region_dict[region_idx][f_c]['weight'] += score[k]
                    region_dict[region_idx][f_c]['sel'].append(node_list_all[k])

    max_value = np.amax(base_img) + 1
    for item in region_dict:
        if len(region_dict[item]) > 1:
            if region_dict[item][0]['weight'] < region_dict[item][1]['weight']:
                frame = 1
            else:
                frame = 0
        else:
            frame = list(region_dict[item].keys())[0]
        selected = region_dict[item][frame]['sel']
        for sel in selected:
            for s in sel:
                base_img[watershed_img == s] = max_value
            max_value += 1

    imsave(result_name, np.uint16(base_img))


def ref_can_matching(list_ref, list2, list2_idx, wsr):
    ## find the ref nodes which have the overlap with the list of candidates

    if list_ref is None:
        return [], 0

    ## for each ref node, find the maximum IoU combination, and then select the combination which has the
    # maximum IoU
    weight = []
    recorder = [[], []]
    idxes = np.unique(list_ref)
    idxes = idxes[idxes>0]

    sel, scores = np.zeros(len(list2)), np.zeros(len(list2))

    ### find possible matching pairs
    for k, node2 in enumerate(list2):
        set2 = set(node2)
        for j, idx in enumerate(idxes):
            node1 = np.ravel_multi_index(np.where(list_ref==idx), list_ref.shape)
            set1 = set(node1)
            intersection = float(len(set2.intersection(set1)))
            iou = intersection/len(set2.union(set1))
            diff = abs(len(set2) - len(set1)) / len(set1)
            # print(iou, diff)
            if iou > 0.2 and diff < args.size_diff:
                weight.append(iou)
                recorder[0].append(j)
                recorder[1].append(k)

    if len(weight) > 0:
        sBranch = len(list_ref)
        tBranch = len(wsr)
        Branch = sBranch + tBranch

        ### construct constraints
        ### Aeq: mxn, m branches and n matching pairs
        weight = np.array(weight)
        recorder = np.array(recorder)
        selectlist = np.array([t for t in range(tBranch)])

        Aeq = np.zeros((Branch, len(weight)))
        Beq = np.ones(Branch)
        for j in range(len(weight)):
            sIdx = recorder[0][j]
            tIdx = recorder[1][j]
            ### branches in source
            Aeq[sIdx][j] = 1

            ### branches in target
            bralist = [tIdx in sublist for sublist in wsr]
            newlist = selectlist[bralist] + sBranch

            for c in newlist:
                Aeq[c][j] = 1

        X = cp.Variable(len(weight), boolean=True)
        constraints = [Aeq @ X <= Beq]

        weight1 = np.ones(len(weight))*100

        prob = cp.Problem(cp.Maximize(weight.T @ X + weight1.T @ X), constraints)
        prob.solve('GLPK_MI')

        for idx in range(len(X.value)):
            if X.value[idx] > 0:
                sel[recorder[1][idx]] = 1
                scores[recorder[1][idx]] = weight[idx]

    return sel, scores

if __name__ == "__main__":

      parser = ArgumentParser()
      parser.add_argument('--img_dir', type=str, default="./data/")
      parser.add_argument('--watershed_dir', type=str, default="")
      parser.add_argument('--watershed_map_dir', type=str, default="")
      parser.add_argument('--watershed_line_dir', type=str, default="")
      parser.add_argument('--reliable_file', type=str, default="")
      parser.add_argument('--base_img', type=str, default='', help='the first starting frame directory')
      parser.add_argument('--min_region', type=int, default=30)
      parser.add_argument('--min_thres', type=int, default=127)
      parser.add_argument('--low_thres', type=float, default=0.1)
      parser.add_argument('--high_thres', type=float, default=0.9)
      parser.add_argument('--reach', type=int, default=25, help='distance hard cut')
      parser.add_argument('--iter', type=int, default=5)
      parser.add_argument('--rmEMD', type=int, default=0)
      parser.add_argument('--start', type=int, default=0)
      parser.add_argument('--get_base', type=int, default=1)
      parser.add_argument('--size_diff', type=float, default=0.50)
      parser.add_argument('--left_pro', type=int, default=1)
      parser.add_argument('--right_pro', type=int, default=1)
      parser.add_argument('--valid_start', type=int, default=0)
      parser.add_argument('--valid_stop', type=int, default=0)

      args = parser.parse_args()
      print(args)
      main(args)

