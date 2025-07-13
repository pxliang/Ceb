import numpy as np
import glob
import os
import pickle
import cv2
import csv
from collections import deque, defaultdict
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.transform import resize
from argparse import ArgumentParser
from scipy.ndimage import grey_dilation, grey_erosion
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import generic_filter
from skimage.morphology import medial_axis, remove_small_objects
from sklearn.linear_model import LinearRegression
import math
import multiprocessing

def find_neighbors(points, h, w):

    offset = [-1, 0, 1]
    y, x = points
    nei = []
    for i in offset:
        for j in offset:
            cur_y, cur_x = y+i, x+j
            if cur_y >= 0 and cur_y < h and cur_x >= 0 and cur_x < w and not ( cur_y == y and cur_x == x ):
                nei.append((cur_y, cur_x))

    return nei

def find_endpoints(points, h, w):

    coord_to_idx = defaultdict(int)

    # step 1: give index of point nodes
    for k in range(len(points)):
        coord_to_idx[(points[k][0], points[k][1])] = k

    # step 2: build node graphs
    graph = defaultdict(set)
    for k, p in enumerate(points):
        neighbors = find_neighbors(p, h, w)
        for n in neighbors:
            if n in coord_to_idx:
                graph[k].add(coord_to_idx[n])

    ## apply Floyd-Wasrshall algorithm
    matrixes = np.ones((len(points), len(points)))*np.inf
    for node in graph:
        for e in graph[node]:
            matrixes[node][e] = 1

    for i in range(len(graph)):
        matrixes[i][i] = 0

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                if matrixes[i][j] > matrixes[i][k] + matrixes[k][j]:
                    matrixes[i][j] = matrixes[i][k] + matrixes[k][j]

    max_dis, ep = 0, ()
    for i in range(len(graph)):
        for j in range(len(graph)):
            if matrixes[i][j] > max_dis:
                max_dis = matrixes[i][j]
                ep = (i, j)

    if len(ep) == 0:
        return None, None

    else:
        return points[ep[0]], points[ep[1]]

def get_fore_back_boundary(img_binary, ws_region):
    contours, hirec = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result = np.zeros(img_binary.shape, dtype = np.uint16)

    for item in contours:
        points = np.array(item).T
        result[points[1], points[0]] = 1

    fore_boundary = ws_region.copy()
    fore_boundary[result == 0] = 0

    return np.uint16(fore_boundary)


def build_line_graph(fore_boundary, ws_boundary_pad, ws_line):
    """

    :param fore_boundary:
    :param ws_boundary_pad:
    :param ws_line: line: {{'value': }, {'label': }}
    :return: line_dict: a dictionary: line:{{'region': }, {'neighbor': }}
    """

    line_dict = defaultdict(dict)

    unique_idx = np.unique(fore_boundary)
    unique_idx = unique_idx[unique_idx > 0]
    ## a dictionary record the mapping between ins to line, ins:line
    ins_to_line = defaultdict(set)
    line = set()

    ## process foreground lines
    for idx in unique_idx:
        line.add((0, idx))
        ins_to_line[idx].add((0, idx))
        points = np.array(np.where(fore_boundary==idx)).T
        line_dict[(0, idx)]['region'] = points
        line_dict[(0, idx)]['length'] = len(points)
        line_dict[(0, idx)]['neighbor'] = set()
        line_dict[(0, idx)]['fore_value'] = idx

    ## process ws lines
    for l in ws_line:
        idx = ws_line[l]['value']
        line.add(l)
        ins_to_line[l[0]].add(l)
        ins_to_line[l[1]].add(l)
        points = np.array(np.where(ws_boundary_pad==idx)).T
        line_dict[l]['region'] = points
        line_dict[l]['length'] = len(points)
        line_dict[l]['neighbor'] = set()
        line_dict[l]['ws_value'] = idx
        line_dict[l]['label'] = ws_line[l]['label']

    ## for each line, final all its neighbors
    for item in line:
        for n in item:
            if n > 0:
                for nei in ins_to_line[n]:
                    if nei != item:
                        line_dict[item]['neighbor'].add(nei)

    return line_dict

def get_sampled_points(item, points, max_point):

    points = np.array(points)

    dis = np.linalg.norm(item - points, axis=1)
    idx_dis = np.argsort(dis)
    dis.sort()
    dis = np.array(dis)
    dis = dis[dis < max_point]
    neighbor_points = idx_dis[:min(max_point, len(dis))]


    return points[neighbor_points]

def get_nearest_line(point, neighbors, line_dict):
    min_dis = []
    min_dis_coord = []
    neighbors = list(neighbors)
    if len(neighbors) < 2:
        print('No neighbors')
        return None

    for n in neighbors:
        line_points = line_dict[n]['region']
        dis = np.linalg.norm(line_points - point, axis=1)
        min_dis.append(np.amin(dis))
        min_dis_coord.append(line_points[np.argmin(dis)])

    # sort neighbor lines by distance.
    sort_idx = np.argsort(min_dis)

    if min_dis[sort_idx[1]] > 10:
        return None

    if len(neighbors) > 2:
        assert min_dis[sort_idx[2]] >= 1

    return [(neighbors[sort_idx[0]], min_dis_coord[sort_idx[0]], min_dis[sort_idx[0]]),
            (neighbors[sort_idx[1]], min_dis_coord[sort_idx[1]], min_dis[sort_idx[1]])]


def point_to_angle(pointA, pointB, pointC):

    BA = pointA - pointB
    BC = pointC - pointB

    cosine_angle = np.clip(np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC)), -1, 1)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def process_line(line_dict,line, max_point):

    points = line_dict[line]['endpoints']

    sampled_points, distances, endpoints, angles_all = [], [], [], []

    if len(points) == 0:
        return [], []

    # print('the line has {} endpoints'.format(len(points)))
    for item in points:
        lines = get_nearest_line(item, line_dict[line]['neighbor'], line_dict)

        if lines is not None:
            line1, point1, dis1, line2, point2, dis2 = lines[0][0], lines[0][1], lines[0][2], lines[1][0], lines[1][1], lines[1][2]

            sampled_points_ws = get_sampled_points(item,  line_dict[line]['region'], max_point)
            sampled_points_nei1 = get_sampled_points(point1, line_dict[line1]['region'], max_point)
            sampled_points_nei2 = get_sampled_points(point2, line_dict[line2]['region'], max_point)

            sampled_points.append([sampled_points_ws, sampled_points_nei1, sampled_points_nei2])
            distances.append(dis2)
            endpoints.append(item)

    if len(sampled_points) < 2:
        return sampled_points, endpoints

    else:
        # sort neighbor lines by distance.
        sort_idx = np.argsort(distances)

        assert distances[sort_idx[1]] <= 20

        return [sampled_points[sort_idx[0]], sampled_points[sort_idx[1]]], [endpoints[sort_idx[0]], endpoints[sort_idx[1]]]


class Para(object):
    def __init__(self, img_name, ws_boundary_name, ws_region_name, ws_map_name, max_point, thres, pos_dir, neg_dir, unknown_dir):
        self.img_name = img_name
        self.ws_boundary_name = ws_boundary_name
        self.ws_region_name = ws_region_name
        self.ws_map_name = ws_map_name
        self.thres = thres
        self.max_point = max_point
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.unknown_dir = unknown_dir


def main(args):

    prob_dir = args.prob_dir.split(',')
    ws_boundary_dir = args.ws_boundary_dir.split(',')
    final_dir = args.final_dir.split(',')
    ws_region_dir = args.ws_region_dir.split(',')
    ws_map_dir = args.ws_map_dir.split(',')
    append = args.append.split(',')

    # print(prob_dir, ws_boundary_dir, final_dir, ws_region_dir, ws_map_dir, append)
    for count in range(len(prob_dir)):

        pos_dir = os.path.join(final_dir[count], '1')
        neg_dir = os.path.join(final_dir[count], '0')
        unknown_dir = os.path.join(final_dir[count], '-1')

        os.makedirs(os.path.join(final_dir[count], '1'), exist_ok=True)
        os.makedirs(os.path.join(final_dir[count], '0'), exist_ok=True)
        os.makedirs(os.path.join(final_dir[count], '-1'), exist_ok=True)

        sub_dir = [f for f in os.listdir(prob_dir[count]) if os.path.isdir(os.path.join(prob_dir[count], f))]
        sub_dir.sort()
        if len(sub_dir) == 0:
            sub_dir = ['']

        for item in sub_dir:
            imgname = [f for f in os.listdir(os.path.join(prob_dir[count], item, append[count])) if f.endswith('.png') or f.endswith('.tif')]
            imgname.sort()

            paras = []
            for i in range(len(imgname)):
                img_name = os.path.join(prob_dir[count], item, append[count], imgname[i])
                ws_boundary_name = os.path.join(ws_boundary_dir[count], imgname[i])
                ws_region_name = os.path.join(ws_region_dir[count], imgname[i])
                ws_map_name = os.path.join(ws_map_dir[count], imgname[i][:-4]+'.pickle')

                print(img_name, ws_boundary_name, ws_region_name, ws_map_name, args.max_point, args.thres, pos_dir, neg_dir, unknown_dir)
                para = Para(img_name, ws_boundary_name, ws_region_name, ws_map_name, args.max_point, args.thres, pos_dir, neg_dir, unknown_dir)
                paras.append(para)


            p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            p.map(parallel_GetBoundary, paras)


def parallel_GetBoundary(para_input):

    max_size = 0
    strel = np.zeros((5, 5))
    # img = imread(os.path.join(para_input.prob_dir, item, append[count], imgname[i]))
    img = imread(para_input.img_name)

    img_pad = np.zeros((img.shape[0] + 10, img.shape[1] + 10), dtype = np.uint8)
    img_pad[5:-5, 5:-5] = img
    img_binary = np.uint8(img_pad > para_input.thres)

    # ws_boundary = imread(os.path.join(ws_boundary_dir[count], imgname[i]))
    ws_boundary = imread(para_input.ws_boundary_name)
    ws_boundary_pad = np.zeros((img.shape[0] + 10, img.shape[1] + 10), dtype=np.uint16)
    ws_boundary_pad[5:-5, 5:-5] = ws_boundary

    # ws_region = imread(os.path.join(ws_region_dir[count], imgname[i]))
    ws_region = imread(para_input.ws_region_name)
    ws_region_pad = np.zeros((img.shape[0] + 10, img.shape[1] + 10), dtype=np.uint16)
    ws_region_pad[5:-5, 5:-5] = ws_region

    fore_boundary = get_fore_back_boundary(img_binary, ws_region_pad)

    with open(para_input.ws_map_name, 'rb') as handle:
        ws_line = pickle.load(handle)

    line_dict = build_line_graph(fore_boundary, ws_boundary_pad, ws_line)

    img_sub_temp, label_temp, line_names_temp = [], [], []

    count_line = 0
    for line in line_dict:
        if line[0] > 0:
            im = np.zeros(img_pad.shape, dtype=np.uint8)
            im[line_dict[line]['region'][:, 0], line_dict[line]['region'][:, 1]] = 1

            result = find_endpoints(line_dict[line]['region'], img_pad.shape[0], img_pad.shape[1])

            if result[0] is not None:
                line_dict[line]['endpoints'] = (result[0], result[1])

                if len(np.unique(label(im > 0))) > 2:
                    line_dict[line]['segments'] = len(np.unique(label(im > 0))) - 1
                else:
                    line_dict[line]['segments'] = 1

                if line_dict[line]['segments'] == 1:
                    # print('processing line: ', line[0], line[1])
                    sampled_points, endpoints = process_line(line_dict,line, para_input.max_point)

                    binary_boundary = np.zeros(img_pad.shape, dtype=np.uint8)
                    count_line += 1

                    for three_points in sampled_points:
                        for k, points in enumerate(three_points):
                            if k == 0:
                                binary_boundary[points[:, 0], points[:, 1]] = 200
                            else:
                                binary_boundary[points[:, 0], points[:, 1]] = 100


                    if len(sampled_points) > 0:

                        binary_boundary = grey_dilation(binary_boundary.astype(np.int32),
                                                        structure=strel.astype(np.int8))
                        boundaries = np.where(binary_boundary > 0)
                        y_min, y_max, x_min, x_max = np.amin(boundaries[0]), np.amax(boundaries[0]), np.amin(boundaries[1]), np.amax(boundaries[1])
                        h, w = y_max - y_min, x_max - x_min

                        binary_boundary = binary_boundary[y_min:y_max+1, x_min:x_max+1]

                        img_sub_temp.append(binary_boundary)
                        label_temp.append(line_dict[line]['label'])
                        line_names_temp.append(os.path.basename(para_input.img_name)[:-4]+'_'+str(line[0])+'_'+str(line[1])+'.png')
                        max_size = max(max(max_size, h), w)

    max_size = (max_size//2+1) * 2
    print(max_size)

    with open('max_size.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([str(max_size)])

    for k, item in enumerate(img_sub_temp):
        final_mask = np.zeros((max_size, max_size), dtype = np.uint8)
        h, w = item.shape[0], item.shape[1]
        final_mask[max_size//2 - h//2:max_size//2 - h//2+h, max_size//2-w//2:max_size//2-w//2+w] = item
        if label_temp[k] == 1:
            imsave(os.path.join(para_input.pos_dir, line_names_temp[k]), final_mask)
        elif label_temp[k] == 0:
            imsave(os.path.join(para_input.neg_dir, line_names_temp[k]), final_mask)
        elif label_temp[k] == -1:
            imsave(os.path.join(para_input.unknown_dir, line_names_temp[k]), final_mask)
        else:
            print('invalid label!')
            sys.exit()



if __name__ == "__main__":

      parser = ArgumentParser()
      parser.add_argument('--final_dir', type=str, default="./App/")
      parser.add_argument('--prob_dir', type=str, default="")
      parser.add_argument('--ws_boundary_dir', type=str, default="")
      parser.add_argument('--ws_region_dir', type=str, default="")
      parser.add_argument('--ws_map_dir', type=str, default="")
      parser.add_argument('--append', type=str, default="")
      parser.add_argument('--max_point', type=int, default=20)
      parser.add_argument('--thres', type=int, default=127)
      parser.add_argument('--max_size', type=int, default=0)
      parser.add_argument('--radius', type=int, default=5)

      args = parser.parse_args()
      print(args)
      main(args)
