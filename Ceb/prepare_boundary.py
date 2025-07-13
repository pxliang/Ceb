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

# Line ends filter
def lineEnds(P):
    """Central pixel and just one other must be set to be a line end"""
    mask = np.ones(9, dtype = np.uint8)
    mask[0], mask[2], mask[6], mask[8] = 0, 0, 0, 0
    P = P*mask
    return ((P[4]==1) and np.sum(P)==2)

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
        print('less than two neighbors')
        return None

    for n in neighbors:
        line_points = line_dict[n]['region']
        dis = np.linalg.norm(line_points - point, axis=1)
        min_dis.append(np.amin(dis))
        min_dis_coord.append(line_points[np.argmin(dis)])

    sort_idx = np.argsort(min_dis)

    if min_dis[sort_idx[1]] > 5:
        print('too far: ', min_dis[sort_idx[1]])
        return None
    else:
        if len(neighbors) > 2:
            assert min_dis[sort_idx[2]] >= 1

        return [(neighbors[sort_idx[0]], min_dis_coord[sort_idx[0]]),
                (neighbors[sort_idx[1]], min_dis_coord[sort_idx[1]])]


def process_line(line_dict,line, max_point):

    points = line_dict[line]['endpoints']

    sampled_points = []

    if len(points) < 2:
        print("less than two endpoints: ", len(points))

    if len(points) == 0:
        return sampled_points

    for item in points:
        lines = get_nearest_line(item, line_dict[line]['neighbor'], line_dict)

        if lines is not None:
            line1, point1, line2, point2 = lines[0][0], lines[0][1], lines[1][0], lines[1][1]

            sampled_points_ws = get_sampled_points(item,  line_dict[line]['region'], max_point)
            sampled_points_nei1 = get_sampled_points(point1, line_dict[line1]['region'], max_point)
            sampled_points_nei2 = get_sampled_points(point2, line_dict[line2]['region'], max_point)

            sampled_points.append([sampled_points_ws, sampled_points_nei1, sampled_points_nei2])


    return sampled_points


def main(args):

    prob_dir = args.prob_dir.split(',')
    ws_boundary_dir = args.ws_boundary_dir.split(',')
    final_dir = args.final_dir.split(',')
    ws_region_dir = args.ws_region_dir.split(',')
    ws_map_dir = args.ws_map_dir.split(',')

    print(prob_dir, ws_boundary_dir, final_dir, ws_region_dir, ws_map_dir)

    img_sub_all, label_all, line_names_all, max_size = [], [], [], 0
    pos_dirs, neg_dirs = [], []
    for count in range(len(prob_dir)):

        pos_dirs.append(os.path.join(final_dir[count], '1'))
        neg_dirs.append(os.path.join(final_dir[count], '0'))

        os.makedirs(os.path.join(final_dir[count], '1'), exist_ok=True)
        os.makedirs(os.path.join(final_dir[count], '0'), exist_ok=True)

        sub_dir = [f for f in os.listdir(prob_dir[count]) if os.path.isdir(os.path.join(prob_dir[count], f))]
        sub_dir.sort()
        if len(sub_dir) == 0:
            sub_dir = ['']

        img_sub_temp, label_temp, line_names_temp = [], [], []

        for item in sub_dir:
            imgname = [f for f in os.listdir(os.path.join(prob_dir[count], item)) if f.endswith('.png') or f.endswith('.tif')]
            imgname.sort()

            strel = np.zeros((args.radius, args.radius))
            for i in range(len(imgname)):
                print(imgname[i])
                img = imread(os.path.join(prob_dir[count], item, imgname[i]))

                img_pad = np.zeros((img.shape[0] + 10, img.shape[1] + 10), dtype = np.uint8)
                img_pad[5:-5, 5:-5] = img
                img_binary = np.uint8(img_pad > 127)

                ws_boundary = imread(os.path.join(ws_boundary_dir[count], imgname[i]))
                ws_boundary_pad = np.zeros((img.shape[0] + 10, img.shape[1] + 10), dtype=np.uint16)
                ws_boundary_pad[5:-5, 5:-5] = ws_boundary

                ws_region = imread(os.path.join(ws_region_dir[count], imgname[i]))
                ws_region_pad = np.zeros((img.shape[0] + 10, img.shape[1] + 10), dtype=np.uint16)
                ws_region_pad[5:-5, 5:-5] = ws_region

                fore_boundary = get_fore_back_boundary(img_binary, ws_region_pad)

                with open(os.path.join(ws_map_dir[count], imgname[i][:-4]+'.pickle'), 'rb') as handle:
                    ws_line = pickle.load(handle)

                line_dict = build_line_graph(fore_boundary, ws_boundary_pad, ws_line)


                count_line = 0
                for line in line_dict:
                    if line[0] > 0:
                        im = np.zeros(img_pad.shape, dtype=np.uint8)
                        im[line_dict[line]['region'][:, 0], line_dict[line]['region'][:, 1]] = 1

                        # Find line ends
                        result = generic_filter(im, lineEnds, (3, 3))
                        points = np.array(np.where(result > 0)).T
                        line_dict[line]['endpoints'] = points

                        if len(np.unique(label(im > 0))) > 2:
                            line_dict[line]['segments'] = len(np.unique(label(im > 0))) - 1
                        else:
                            line_dict[line]['segments'] = 1

                        if line_dict[line]['segments'] == 1 and ( line_dict[line]['label'] == 0 or line_dict[line]['label'] == 1 ):
                            sampled_points = process_line(line_dict,line, args.max_point)

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
                                line_names_temp.append(imgname[i][:-4]+'_'+str(line[0])+'_'+str(line[1])+'.png')
                                max_size = max(max(max_size, h), w)

        img_sub_all.append(img_sub_temp)
        label_all.append(label_temp)
        line_names_all.append(line_names_temp)

    max_size = (max_size//2+1) * 2
    print(max_size)

    for count, items in enumerate(img_sub_all):
        for k, item in enumerate(items):
            final_mask = np.zeros((max_size, max_size), dtype = np.uint8)
            h, w = item.shape[0], item.shape[1]
            final_mask[max_size//2 - h//2:max_size//2 - h//2+h, max_size//2-w//2:max_size//2-w//2+w] = item

            if args.resize_size > 0:
                final_mask = final_mask > 0
                final_mask = np.uint8(resize(final_mask, (args.resize_size, args.resize_size), order=0))*255

            if label_all[count][k] > 0:
                imsave(os.path.join(pos_dirs[count], line_names_all[count][k]), final_mask)
            else:
                imsave(os.path.join(neg_dirs[count], line_names_all[count][k]), final_mask)




if __name__ == "__main__":

      parser = ArgumentParser()
      parser.add_argument('--final_dir', type=str, default="./App/")
      parser.add_argument('--prob_dir', type=str, default="")
      parser.add_argument('--ws_boundary_dir', type=str, default="")
      parser.add_argument('--ws_region_dir', type=str, default="")
      parser.add_argument('--ws_map_dir', type=str, default="")
      parser.add_argument('--resize_size', type=int, default=0)
      parser.add_argument('--max_point', type=int, default=20)
      parser.add_argument('--radius', type=int, default=5)

      args = parser.parse_args()
      print(args)
      main(args)
