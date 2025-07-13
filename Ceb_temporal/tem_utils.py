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

from itertools import permutations, combinations

def GetComFast_Tree(img, min_thres, min_cc):

    '''
    :param img:
    :param min_thres:
    :param min_cc:
    :return: out: CxNxM (C: # of threshold values, N, M: image shape)
             node_dict: node list
             threshold: threshold value list
    '''
    ### get threshold value list
    thresholds = np.unique(img)
    thresholds = thresholds[thresholds > min_thres]

    thresholds.sort()
    num_th = len(thresholds)
    if num_th < 1:
        return [], [], []
    else:

        ### get labeled images
        all_img = np.repeat(img[:, :, np.newaxis], num_th, axis=2)
        pb_map = all_img >= thresholds
        out = np.array([label(remove_small_objects(pb_map[:, :, i], min_cc)) for i in range(num_th)])

        ### build forest
        node_dict = OrderedDict()

        ### get roots
        temp = out[0]
        #region = regionprops(temp)
        test = np.unique(temp[temp > 0])
        test = np.sort(test)

        for index in test:
            #print(index)
            node_dict[(0, index)] = Node((0, index))

        ### find edges
        if num_th > 1:
            test_parents = out[:num_th-1, :, :]
            test_childs = out[1:, :, :]
            edge_list = list()
            for test_parent, test_child in zip(test_parents, test_childs):
                mask = test_child > 0
                parent_mask = test_parent[mask]
                child_mask = test_child[mask]
                combine = np.stack([parent_mask, child_mask])
                if len(combine[0]) > 0:
                    edge = np.unique(combine, axis=1)
                    edge_list.append(edge)

            # for each edge, start check whether exist node ends with, if so, append the path, if not, build a new one
            for index, edges in enumerate(edge_list):
                for edge_index in range(edges.shape[1]):
                    parent, child = edges[:, edge_index]
                    parent_id = (index, parent)
                    child_id = (index + 1, child)
                    node_dict[child_id] = Node(child_id, parent=node_dict[parent_id])

        #print('length: ', len(node_dict))
        root = []
        for node in node_dict:
            #print('node: ', node)
            if node_dict[node].is_root:
                root.append(node_dict[node])

        new_dict = OrderedDict()
        for node in root:
            #print([n for n in PreOrderIter(node)])
            f = delete(node, True)[0]

            for n in PreOrderIter(f):
                new_dict[n.name] = n
                assert len(n.children) != 1

        nameToIdx = defaultdict(int)
        for i, n in enumerate(new_dict):
            #print('n: ', n)
            nameToIdx[n] = i
        path = GetPath(new_dict)
        path_all = []
        for p in path:
            path_all.append([nameToIdx[l] for l in p])
        return out, new_dict, path_all

def delete(root, exist):

    Flag = True if len(root.children) > 1 else False
    if exist:
        new_child = []
        for n in root.children:
            new_child.extend(delete(n, Flag))
        root.children = new_child
        return [root]

    else:
        node = []
        for n in root.children:
            node.extend(delete(n, Flag))
        return node

def GetComFast_valid(img, gt, graph):

    ### get watershed value list
    wsr = np.unique(img)
    wsr = wsr[wsr > 0]
    wsr.sort()

    gt_idx = np.unique(gt)
    gt_idx = gt_idx[gt_idx > 0]

    if len(wsr) < 1:
        return [], []
    else:
        node_list, including_list = set(), []
        ### get labeled images

        for idx in gt_idx:
            cover = np.unique(img[gt == idx])
            cover = cover[cover > 0]
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
                        node_list.add(item)

        node_list = list(node_list)
        for idx in wsr:
            temp = []
            for t, node in enumerate(node_list):
                if idx in node:
                    temp.append(t)
            including_list.append(temp)
        return including_list, node_list

def GetComFast(img):
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
        for i in range(len(wsr)):
            node_list.extend(combinations(wsr, i+1))

        for idx in wsr:
            temp = []
            for t, node in enumerate(node_list):
                if idx in node:
                    temp.append(t)
            including_list.append(temp)
        return including_list, node_list


def get_canddiates(watershed_img, pm_img, base_img):

    ## get the list of candidates in n*h*w, each h*w image represents a list of candidates under a 0.5 node
    ## the list of candidates are labeled by the same number in the watershed_img
    temp = watershed_img
    temp[base_img > 0] = 0
    pm_img = label(pm_img > 127)
    unique_idx = np.unique(pm_img)
    unique_idx = unique_idx[unique_idx > 0]

    out = []
    for idx in unique_idx:
        overlap = np.unique(temp[pm_img == idx])
        overlap = overlap[overlap > 0]
        if len(overlap) > 0:
            sub_img = np.zeros(temp.shape, dtype=np.uint16)
            for o_idx in overlap:
                sub_img[temp == o_idx] = o_idx
                out.append(sub_img)

    return np.array(out)


def get_sub_region(watershed_img, pm_img):
    ## get the list of candidates in n*h*w, each h*w image represents a list of candidates under a 0.5 node
    ## the list of candidates are labeled by the same number in the watershed_img
    pm_img = label(pm_img > 127)
    unique_idx = np.unique(pm_img)
    unique_idx = unique_idx[unique_idx > 0]

    out = []
    for idx in unique_idx:
        overlap = np.unique(watershed_img[pm_img == idx])
        overlap = overlap[overlap > 0]
        if len(overlap) > 0:
            out.append(overlap)

    idx_to_region = defaultdict(int)
    for k, item in enumerate(out):
        for i in item:
            idx_to_region[i] = k

    return idx_to_region


def GetPath(node_dict):
    ### Given a forest, return all paths
    leaves = []
    path = []
    for node in node_dict:
        if node_dict[node].is_leaf:
            leaves.append(node_dict[node])
    for node in leaves:
        path.append([nd.name for nd in node.path])

    return path


def get_ref(img, base_img):

    img = EMD_match(base_img, img)

    return img

def get_ref_noEMD(img, base_img):

    ## use EMD to remove redundant matching
    # img = EMD_match(base_img, img)
    index = np.unique(img)
    index = index[index > 0]

    list_all = []
    for node in index:
        cell = np.where(img == node)
        list_all.append(tuple(cell))

    treeIdx = [i for i in range(len(list_all))]

    return list_all, treeIdx

def dynamic_IoU(cell_pred, ref_img, dis, size_diff):

    max_iou = 0
    for dis_y in dis:
        for dis_x in dis:
            if dis_y == 0:
                y1s, y1e = 0, ref_img.shape[0]
                y2s, y2e = 0, ref_img.shape[0]
            elif dis_y > 0:
                y1s, y1e = dis_y, ref_img.shape[0]
                y2s, y2e = 0, -dis_y
            else:
                y1s, y1e = 0, dis_y
                y2s, y2e = -dis_y, ref_img.shape[0]

            if dis_x == 0:
                x1s, x1e = 0, ref_img.shape[1]
                x2s, x2e = 0, ref_img.shape[1]
            elif dis_x > 0:
                x1s, x1e = dis_x, ref_img.shape[1]
                x2s, x2e = 0, -dis_x
            else:
                x1s, x1e = 0, dis_x
                x2s, x2e = -dis_x, ref_img.shape[1]

            temp_cur = np.zeros(ref_img.shape)
            temp_cur[y1s:y1e, x1s:x1e] = ref_img[y2s:y2e, x2s:x2e]

            cell_ref = set(np.ravel_multi_index(np.where(temp_cur == 1), ref_img.shape))

            if len(cell_ref) > 0:

                iou = len(cell_ref.intersection(cell_pred)) / len(cell_ref.union(cell_pred))
                diff = abs(len(cell_pred) - len(cell_ref)) / len(cell_ref)
                if iou > max_iou and diff < size_diff:
                    max_iou = iou

    return max_iou