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

from matching_utils import get_ref_noEMD, GetComFast, debug

from itertools import combinations

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

    watershed_region_dir = os.path.join(args.watershed_dir, 'region')
    watershed_line_dir = os.path.join(args.watershed_dir, 'map')
    sub_dir = [f for f in os.listdir(args.pm_dir) if os.path.isdir(os.path.join(args.pm_dir, f))]
    sub_dir.sort()
    if len(sub_dir) == 0:
        sub_dir = ['']

    pmname = []
    for item in sub_dir:
        pmname.extend([os.path.join(args.pm_dir, item, f) for f in os.listdir(os.path.join(args.pm_dir, item)) if
                  f.endswith('.png') or f.endswith('.tif')])


    result_dir = os.path.join(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    imgname = [f for f in os.listdir(os.path.join(watershed_region_dir)) if f.endswith('.png') or f.endswith('.tif')]
    refname = [f for f in os.listdir(os.path.join(args.ref_dir)) if f.endswith('.png') or f.endswith('.tif')]

    imgname.sort()
    refname.sort()
    pmname.sort()

    for i in range(len(imgname)):

        print(imgname[i], pmname[i])
        assert imgname[i][:-4] == os.path.basename(pmname[i])[:-4]
        result_name = os.path.join(result_dir, imgname[i][:-4]+'.pickle')
        HEMD(os.path.join(watershed_region_dir, imgname[i]), os.path.join(args.ref_dir, refname[i]),
             pmname[i], os.path.join(watershed_line_dir, imgname[i][:-4]+'.pickle'), result_name)


def HEMD(img_name, ref_name, pm_name, line_name, result_name):

    # current image
    img = imread(img_name)
    pm = imread(pm_name)

    with open(line_name, 'rb') as handle:
        lines = pickle.load(handle)

    _, line_to_label = get_ins_to_line(lines)

    if len(lines) == 0:
        with open(result_name, 'wb') as handle:
            pickle.dump(line_to_label, handle)

    else:
        graph = defaultdict(set)
        for item in lines:
            n1, n2 = lines[item]
            graph[n1].add(n2)
            graph[n2].add(n1)

        ref_img = imread(ref_name)
        # ref_img[img == 0] = 0
        wsr, dict2 = GetComFast(img, pm, ref_img, graph)

        ref_img_reorder = np.zeros(ref_img.shape, dtype = np.uint16)
        unique_idx = np.unique(ref_img)
        unique_idx = unique_idx[unique_idx > 0]
        for k, idx in enumerate(unique_idx):
            ref_img_reorder[ref_img == idx] = k+1

        list2 = []
        for node in dict2:
            cell_temp = []
            for n in node:
                cell = np.where(img == n)
                minx, maxx, miny, maxy = min(cell[0]), max(cell[0]), min(cell[1]), max(cell[1])
                if maxx > minx and maxy > miny:
                    cell_temp.extend(np.ravel_multi_index(cell, img.shape, order='C'))
            list2.append(cell_temp)

        cur_sel, pairs = ref_can_matching(ref_img_reorder, list2, wsr)

        # debug(pairs, img, dict2, ref_img_reorder)

        inter_edge, external_edge = set(), set()
        for i in cur_sel:
            com = dict2[i]
            for node in com:
                for e in graph[node]:
                    if e in com:
                        inter_edge.add((min(node, e), max(node, e)))
                    else:
                        external_edge.add((min(node, e), max(node, e)))

        for edge in inter_edge:
            line_to_label[edge]['label'] = 0
        for edge in external_edge:
            line_to_label[edge]['label'] = 1

        with open(result_name, 'wb') as handle:
            pickle.dump(line_to_label, handle)


def ref_can_matching(ref_img, list2, wsr):
    weight, scale = [], []
    recorder = [[], []]

    ref_img_flat = ref_img.reshape(ref_img.shape[0]*ref_img.shape[1], order='C')
    sel = []
    ### find possible matching pairs
    for k, node2 in enumerate(list2):
        set2 = set(node2)
        list_ref = np.unique(ref_img_flat[node2])
        list_ref = list_ref[list_ref > 0]
        for j in list_ref:
            node1 = np.where(ref_img_flat == j)[0]
            set1 = set(node1)
            # size_change = float(abs(len(node1)-len(node2)))/len(node1)
            # IoU = float(len(set2.intersection(set1))) / len(set2.union(set1))
            IoU = float(len(set2.intersection(set1))) / len(set2.union(set1))

            if IoU > 0.2:
               weight.append(IoU)
               recorder[0].append(j)
               recorder[1].append(k)

    if len(weight) > 0:
        unique_gt = np.unique(ref_img)
        unique_gt = unique_gt[unique_gt > 0]
        sBranch = len(unique_gt)
        tBranch = len(wsr)
        Branch = sBranch + tBranch

        ### construct constraints
        ### Aeq: mxn, m branches and n matching pairs
        print('length of weight: ', len(weight))
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

        X = cp.Variable(len(weight), boolean = True)
        constraints = [Aeq@X<=Beq]

        prob = cp.Problem(cp.Maximize(weight.T@X), constraints)
        prob.solve('GLPK_MI')

        pairs = []
        for idx in range(len(X.value)):
            if X.value[idx] > 0:
                sel.append(recorder[1][idx])
                pairs.append((recorder[0][idx], recorder[1][idx]))

        # print('selected: ', sel)

    return sel, pairs

if __name__ == "__main__":

      parser = ArgumentParser()
      parser.add_argument('--ref_dir', type=str, default="./data/")
      parser.add_argument('--pm_dir', type=str, default="./data/")
      parser.add_argument('--result_dir', type=str, default="./App/")
      parser.add_argument('--watershed_dir', type=str, default="")
      parser.add_argument('--min_region', type=int, default=30)

      args = parser.parse_args()
      print(args)
      main(args)

