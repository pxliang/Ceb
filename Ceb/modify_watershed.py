import os
import argparse
import numpy as np
from collections import deque, defaultdict

import sys
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage import img_as_float32
from skimage.morphology import remove_small_objects

import pickle

from scipy.io import loadmat

# Implementation of:
# Pierre Soille, Luc M. Vincent, "Determining watersheds in digital pictures via
# flooding simulations", Proc. SPIE 1360, Visual Communications and Image Processing
# '90: Fifth in a Series, (1 September 1990); doi: 10.1117/12.24211;
# http://dx.doi.org/10.1117/12.24211
class Watershed(object):
   MASK = -2
   WSHD = 0
   INIT = -1
   INQE = -3

   def __init__(self, levels = 256):
      self.levels = levels

   # Neighbour (coordinates of) pixels, including the given pixel.
   def _get_neighbors(self, height, width, pixel):
      return np.mgrid[
         max(0, pixel[0] - 1):min(height, pixel[0] + 2),
         max(0, pixel[1] - 1):min(width, pixel[1] + 2)
      ].reshape(2, -1).T

   def apply(self, image, marker, min_thres):
      flag = False
      fifo = deque()

      # set 0 to remove backgroud pixels and marker pixels during the watershed process
      image[marker > 0] = 0
      image[image > -min_thres] = 0
      height, width = image.shape
      total = height * width
      labels = marker.copy().astype(np.int32)
      labels[labels == 0] = self.INIT

      reshaped_image = image.reshape(total)
      # [y, x] pairs of pixel coordinates of the flattened image.
      pixels = np.mgrid[0:height, 0:width].reshape(2, -1).T
      # Coordinates of neighbour pixels for each pixel.
      neighbours = np.array([self._get_neighbors(height, width, p) for p in pixels])
      if len(neighbours.shape) == 3:
         # Case where all pixels have the same number of neighbours.
         neighbours = neighbours.reshape(height, width, -1, 2)
      else:
         # Case where pixels may have a different number of pixels.
         neighbours = neighbours.reshape(height, width)

      # get the valid list
      ws_indices = np.where(reshaped_image < 0)[0]

      reshaped_image = reshaped_image[ws_indices]
      pixels = pixels[ws_indices]

      indices = np.argsort(reshaped_image)

      ## get the sorted pixels values
      sorted_image = reshaped_image[indices]

      ## get the sorted pixel indices
      sorted_pixels = pixels[indices]
      boundaries = np.zeros(image.shape, dtype=np.int32)
      line_to_ins = defaultdict(tuple)

      if len(sorted_pixels) == 0:
         return labels, boundaries, line_to_ins
      else:

         # self.levels evenly spaced steps from minimum to maximum.
         # levels is to set threshold values, avoiding too many layers, like 0.2581, 0.2589
         levels = np.linspace(sorted_image[0], sorted_image[-1], self.levels)
         level_indices = []
         current_level = 0

         # Get the indices that deleimit pixels with different values.
         for i in range(len(ws_indices)):
            if sorted_image[i] > levels[current_level]:
                  # Skip levels until the next highest one is reached.
                  while sorted_image[i] > levels[current_level]: current_level += 1
                  level_indices.append(i)

         level_indices.append(len(ws_indices))
         start_index = 0

         pixel_to_ins, ins_to_pixel = defaultdict(list), defaultdict(list)
         for stop_index in level_indices:
            # Mask all pixels at the current level.
            for p in sorted_pixels[start_index:stop_index]:
               labels[p[0], p[1]] = self.MASK
               # Initialize queue with neighbours of existing basins at the current level.
               for q in neighbours[p[0], p[1]]:
                  # p == q is ignored here because labels[p] < WSHD
                  if labels[q[0], q[1]] >= self.WSHD:
                     labels[p[0], p[1]] = self.INQE
                     fifo.append(p)
                     break

            assert len(fifo) > 0
            # Extend basins.
            while fifo:
               p = fifo.popleft()
               # Label p by inspecting neighbours.
               for q in neighbours[p[0], p[1]]:
                  # Don't set lab_p in the outer loop because it may change.
                  lab_p = labels[p[0], p[1]]
                  lab_q = labels[q[0], q[1]]
                  if lab_q > 0:
                     if lab_p == self.INQE or (lab_p == self.WSHD and flag):
                        if lab_p == self.WSHD and flag:
                           del pixel_to_ins[tuple(p)]
                        labels[p[0], p[1]] = lab_q

                     elif lab_p > 0 and lab_p != lab_q:
                        labels[p[0], p[1]] = self.WSHD
                        pixel_to_ins[tuple(p)] = (min(lab_p, lab_q), max(lab_p, lab_q))
                        flag = False
                  elif lab_q == self.WSHD:
                     if lab_p == self.INQE:
                        labels[p[0], p[1]] = self.WSHD
                        pixel_to_ins[tuple(p)] = pixel_to_ins[tuple(q)]
                        flag = True
                  elif lab_q == self.MASK:
                     labels[q[0], q[1]] = self.INQE
                     fifo.append(q)

            start_index = stop_index

         for item in pixel_to_ins:
            ins_to_pixel[pixel_to_ins[item]].append(item)

         for k, item in enumerate(ins_to_pixel):
            cell = np.array(ins_to_pixel[item]).T
            boundaries[cell[0], cell[1]] = k+1
            line_to_ins[k+1] = item

         labels[labels == self.INIT] = 0
         # print(line_to_ins)
         return labels, boundaries, line_to_ins



def main(args):

   file_names = [f for f in os.listdir(os.path.join(args.prob_processed_dir)) if f.endswith('.png') or f.endswith('.m')]
   file_names.sort()
   w = Watershed()

   os.makedirs(os.path.join(args.region_dir), exist_ok=True)
   os.makedirs(os.path.join(args.line_dir), exist_ok=True)
   os.makedirs(os.path.join(args.map_dir), exist_ok=True)

   for name in file_names:
      print(name)
      marker = imread(os.path.join(args.marker_dir, name[:-2]+'.png'))
      prob_processed = loadmat(os.path.join(args.prob_processed_dir, name[:-2]+'.m'))['prob']
      labels, boundaries, line_to_ins = w.apply(prob_processed, marker, args.min_thres)

      final_boundary = defaultdict(list)
      for item in line_to_ins:
         if item in boundaries:
            final_boundary[item] = line_to_ins[item]

      with open(os.path.join(args.map_dir, name[:-2]+'.pickle'), 'wb') as handle:
         pickle.dump(final_boundary, handle)
      imsave(os.path.join(args.region_dir, name[:-2]+'.png'), np.uint16(labels))
      imsave(os.path.join(args.line_dir, name[:-2] + '.png'), np.uint16(boundaries))


if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("--prob_processed_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--marker_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--region_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--line_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--map_dir", default="E:/PyTorch/data/coco2017")
   parser.add_argument("--min_cc", type=int, default=10)
   parser.add_argument("--min_thres", type=float, default=0.5)
   args = parser.parse_args()

   os.makedirs(args.region_dir, exist_ok=True)
   os.makedirs(args.line_dir, exist_ok=True)
   os.makedirs(args.map_dir, exist_ok=True)
   main(args)


