import os
import argparse
import numpy as np
from skimage.io import imread, imsave
from collections import defaultdict
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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

   ws_imgs = [f for f in os.listdir(args.img_dir) if f.endswith('.png') or f.endswith('.tif')]
   ws_imgs.sort()

   ws_line = [f for f in os.listdir(args.ws_line_dir) if f.endswith('.png') or f.endswith('.tif')]
   ws_line.sort()

   ws_map = [f for f in os.listdir(args.ws_map_dir) if f.endswith('.pickle')]
   ws_map.sort()


   text_file = open(args.pred_file, "r")
   negative_list = text_file.readlines()

   img_to_neglines = defaultdict(set)

   for item in negative_list:
       item = item.rstrip('\n').split('_')
       img_name, ws_idx1, ws_idx2 = '_'.join(item[:-2]), item[-2], item[-1]
       img_to_neglines[img_name].add((int(ws_idx1), int(ws_idx2)))

   for i, name in enumerate(ws_imgs):
      img = imread(os.path.join(args.img_dir, name))
      result_img = np.zeros(img.shape, dtype = np.uint16)
      with open(os.path.join(args.ws_map_dir, ws_map[i]), 'rb') as handle:
         lines = pickle.load(handle)

      _, line_to_label = get_ins_to_line(lines)

      line_img = imread(os.path.join(args.ws_line_dir, ws_line[i]))

      graph = defaultdict(set)
      unique_idx = np.unique(img)
      unique_idx = unique_idx[unique_idx>0]

      for idx in unique_idx:
         graph[idx] = set()

      negative_list = img_to_neglines[name[:-4]]

      for item in negative_list:
         n1, n2 = item
         graph[n1].add(n2)
         graph[n2].add(n1)

      visit = set()
      count = 1
      for node in graph:
         connected_node, connected_edge = [], []
         if node not in visit:
            stack = [node]
            visit.add(node)
            connected_node.append(node)
            while stack:
               cur = stack.pop()
               for child in graph[cur]:
                  if child not in visit:
                     connected_edge.append((min(cur, child), max(cur, child)))
                     connected_node.append(child)
                     stack.append(child)
                     visit.add(child)

         for node in connected_node:
            result_img[img == node] = count


         for edge in connected_edge:
            # print(line_to_label[edge]['value'])
            result_img[line_img == line_to_label[edge]['value']] = count

         if len(connected_node) != 0:
            count += 1

      imsave(os.path.join(args.output_dir, name), result_img)



if __name__ == "__main__":

   parser = argparse.ArgumentParser()

   parser.add_argument("--img_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--pred_file", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--ws_line_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--ws_map_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--output_dir", default="E:/PyTorch/data/coco2017")
   args = parser.parse_args()

   os.makedirs(args.output_dir, exist_ok=True)
   main(args)
