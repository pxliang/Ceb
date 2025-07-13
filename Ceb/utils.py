import os
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.morphology import thin, remove_small_objects
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle, LevelOrderIter
from skimage.io import imread, imsave, use_plugin
from skimage.morphology import thin, dilation
from collections import defaultdict, OrderedDict, deque


def GetComFast(img, min_thres, min_cc):

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
        test = np.unique(temp[temp > 0])
        test = np.sort(test)

        for index in test:
            node_dict[(0, index)] = Node((0, index), area=len(np.where(out[0]==index)[0]))

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
                    node_dict[child_id] = Node(child_id, parent=node_dict[parent_id], area=len(np.where(out[index+1]==child)[0]))

        #print('length: ', len(node_dict))
        root = []
        for node in node_dict:
            if node_dict[node].is_root:
                root.append(node_dict[node])

        new_dict, new_root = OrderedDict(), []
        for node in root:
            f = delete(node, True)[0]
            f = leaf_max_point(f, img, out)
            # f = prune(f, min_cc)
            # get_combination(f)
            if f is not None:
                new_root.append(f)
                for n in PreOrderIter(f):
                    new_dict[n.name] = n
                    assert len(n.children) != 1

        return out, new_dict, new_root



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

def prune(root, min_size):

    if root.area < min_size:
        return None
    else:
        queue = deque()
        for n in root.children:
            if n.area > min_size:
                queue.append(n)
            else:
                n.parent = None

        while queue:
            node = queue.popleft()
            for n in node.children:
                if n.area > min_size:
                    queue.append(n)
                else:
                    n.parent = None

        return root

def leaf_max_point(root, img, out):

    for n in PreOrderIter(root):
        if n.is_leaf:
            cell = img[out[n.name[0]] == n.name[1]]
            num = np.amax(cell)
            set1 = set(np.ravel_multi_index(np.where(out[n.name[0]] == n.name[1]), img.shape))
            set2 = set(np.ravel_multi_index(np.where(img == num), img.shape))
            n.max_point = set1.intersection(set2)

    return root

def get_max_point(root, img, out):
    cell = img[out[root.name[0]] == root.name[1]]
    num = np.amax(cell)
    set1 = set(np.ravel_multi_index(np.where(out[root.name[0]] == root.name[1]), img.shape))
    set2 = set(np.ravel_multi_index(np.where(img == num), img.shape))
    root.max_point = set1.intersection(set2)
    root.inter = 1
    root.left_child = 1

    queue = deque()
    queue.append(root)

    while queue:
        node = queue.popleft()
        max_iou = 0
        look_result = []
        for n in node.children:
            cell = img[out[n.name[0]] == n.name[1]]
            num = np.amax(cell)
            set1 = set(np.ravel_multi_index(np.where(out[n.name[0]] == n.name[1]), img.shape))
            set2 = set(np.ravel_multi_index(np.where(img == num), img.shape))
            n.max_point = set1.intersection(set2)

            max_point_p = n.parent.max_point
            iou = len(max_point_p.intersection(n.max_point)) / len(max_point_p)
            n.inter = iou
            look_result.append(iou)

            if iou > max_iou:
                max_iou = iou
                left_node = n.name

            queue.append(n)


        if max_iou > 0:
            for n in node.children:
                if n.name == left_node:
                    n.left_child = 1
                else:
                    n.left_child = 0

    return root

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

def get_combination(root):

    node_list = [node for node in LevelOrderIter(root)][::-1]

    for node in node_list:
        left_results, other_results, all_results = [], [], []

        if len(node.children) == 0:
            node.combine = [[node.name]]

        else:
            for n in node.children:
                if n.left_child:
                    left_results.extend(n.combine)
                else:
                    other_results.append(n.combine)

            for item in left_results:
                all_results.append(item)

            temp_results = left_results.copy()
            for item in other_results:
                new_results = []
                for c1 in temp_results:
                    for c2 in item:
                        new_results.append(c1+c2)

                temp_results = new_results

            all_results.extend(temp_results)
            node.combine = all_results



