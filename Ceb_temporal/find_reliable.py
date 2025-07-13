'''
This code is for the complete tree generation version
'''

import numpy as np
import glob
import os
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import thin, remove_small_objects
from collections import defaultdict
from argparse import ArgumentParser
import time
from datetime import datetime
from tem_utils import GetComFast, GetComFast_Tree
import shutil
import pickle

from scipy.io import loadmat

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def tree_reliable(img_dir, result_dir, min_thres, min_region):
    sub_dir = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
    sub_dir.sort()
    if len(sub_dir) == 0:
        sub_dir = ['']

    for item in sub_dir:
        imgname = [f for f in os.listdir(os.path.join(img_dir, item)) if f.endswith('.png') or f.endswith('.tif')]
        imgname.sort()

        os.makedirs(os.path.join(result_dir, item), exist_ok=True)
        for name in imgname:
            img = imread(os.path.join(img_dir, item, name))
            out1, dict1, treeIdx1 = GetComFast_Tree(img, min_thres, min_region)

            temp = np.zeros(img.shape, dtype=np.uint16)
            count = 0
            for node in dict1:
                # if dict1[node].is_leaf and node[0]+args.min_thres < 127:
                if dict1[node].is_leaf and dict1[node].is_root:
                    count += 1
                    temp[out1[node[0]] == node[1]] = count

            imsave(os.path.join(result_dir, item, name[:-4]+'.tif'), np.uint16(temp))

def get_ins_to_line(value_to_line):
    line_to_label = defaultdict(int)

    for item in value_to_line:
        line_to_label[value_to_line[item]] = item

    return line_to_label

def find_unreliable_node(graph):

    unreliable_node = set()
    graph_copy = graph.copy()
    graph_copy[graph_copy == 1] = 0
    for i, item in enumerate(graph_copy):
        if len(np.where(item > 0)[0]) > 0:
            unreliable_node.add(i)

    for i, item in enumerate(graph_copy.T):
        if len(np.where(item > 0)[0]) > 0:
            unreliable_node.add(i)

    return unreliable_node

def boundary_class_reliable(ws_region_dir, ws_line_dir, ws_map_dir, reliable_file, low_thres, high_thres, result_dir):

    text_file = open(reliable_file, "r")
    reliable_list = text_file.readlines()

    ## get all reliable lines
    node_list = defaultdict(dict)
    for item in reliable_list:
        names, value = item.split(',')[0], float(item.split(',')[1])

        img_name, node1, node2 = names.split('_')[0], names.split('_')[1], names.split('_')[2]
        node_list[img_name][(int(node1), int(node2))] = value

    ## build graph nodes and get ccs
    ws_imgs = [f for f in os.listdir(ws_region_dir) if f.endswith('.png') or f.endswith('.tif')]

    for i, name in enumerate(ws_imgs):
        ## get (ins1, ins2) to line_id mapping
        with open(os.path.join(ws_map_dir, name[:-4]+'.pickle'), 'rb') as handle:
            lines = pickle.load(handle)

        line_to_label = get_ins_to_line(lines)

        ## get line img
        line_img = imread(os.path.join(ws_line_dir, name))

        img = imread(os.path.join(ws_region_dir, name))
        reliable_img = np.zeros(img.shape, dtype=np.uint16)

        all_idx = np.unique(img)
        all_idx = all_idx[all_idx > 0]
        all_idx.sort()
        convert = defaultdict()
        for k, idx in enumerate(all_idx):
            convert[idx] = k

        graph = np.zeros((len(all_idx), len(all_idx)), dtype=np.float32)

        ## set all ws lines as 0.5, as some ws lines don't have classifier scores
        for item in line_to_label:
            graph[convert[item[0]], convert[item[1]]] = 0.5
            graph[convert[item[1]], convert[item[0]]] = 0.5

        ## set ws lines with classifier scores
        for item in node_list[name[:-4]]:
            graph[convert[item[0]], convert[item[1]]] = node_list[name[:-4]][item]
            graph[convert[item[1]], convert[item[0]]] = node_list[name[:-4]][item]

        ## find false boundary reliable lines
        false_boundary_list = set()
        for item in zip(np.where(graph > 0.9)[0], np.where(graph > 0.9)[1]):
            false_boundary_list.add((min(all_idx[item[0]], all_idx[item[1]]), max(all_idx[item[0]], all_idx[item[1]])))

        ## set reliable boundaries
        graph[graph < low_thres] = 0
        graph[graph > high_thres] = 1

        unreliable_nodes = find_unreliable_node(graph)

        graph_copy = graph.copy()
        graph_copy[graph_copy > 0] = 1
        graph_copy = np.uint8(graph_copy)

        graph_copy = csr_matrix(graph_copy)
        n_components, labels = connected_components(csgraph=graph_copy, directed=False, return_labels=True)

        result = []
        for k in range(n_components):
            result.append(np.where(labels == k)[0])

        c = np.amax(reliable_img) + 1
        for item in result:
            ## if any node of the current cc is in unreliable_nodes, the answer is not determined yet,
            ## should not be added
            flag = True
            for k in item:
                if k in unreliable_nodes:
                    flag = False
            if flag:
                for k in item:
                    reliable_img[img == all_idx[k]] = c
                c += 1

        ## padding lines
        for item in false_boundary_list:
            node1, node2 = item[0], item[1]
            ## find the id of the reliable line by chekcing its two neighbor nodes
            ins_id1, ins_id2 = np.unique(reliable_img[img == node1]), np.unique(reliable_img[img == node2])

            if len(ins_id1) == 1 and len(ins_id2) == 1:
                assert ins_id1 == ins_id2
                reliable_img[line_img == line_to_label[item]] = ins_id1[0]

        imsave(os.path.join(result_dir, name[:-4]+'.tif'), np.uint16(reliable_img))


def main(args):

    if args.watershed_dir:
        watershed_reliable(args)
    else:
        tree_reliable(args.img_dir, args.result_dir, args.min_thres, args.min_region)



if __name__ == "__main__":

      parser = ArgumentParser()
      parser.add_argument('--img_dir', type=str, default="./data/")
      parser.add_argument('--watershed_dir', type=str, default="")
      parser.add_argument('--min_thres', type=int, default = 40)
      parser.add_argument('--min_region', type=int, default=30)
      parser.add_argument('--result_dir', type=str, default="./data/")

      args = parser.parse_args()
      print(args)
      main(args)

