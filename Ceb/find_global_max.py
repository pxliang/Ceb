import numpy as np
import glob
import os
import pickle
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from argparse import ArgumentParser
from scipy.ndimage import grey_dilation
from skimage.segmentation import watershed
from scipy.io import loadmat

from utils import GetComFast


def main(args):

    sub_dir = [f for f in os.listdir(args.prob_dir) if os.path.isdir(os.path.join(args.prob_dir, f))]
    sub_dir.sort()
    if len(sub_dir) == 0:
        sub_dir = ['']

    for item in sub_dir:
        imgname = [f for f in os.listdir(os.path.join(args.prob_dir, item)) if f.endswith('.png') or f.endswith('.mat')]
        imgname.sort()

        os.makedirs(os.path.join(args.ws_result_dir, item), exist_ok=True)

        for i in range(len(imgname)):
            if imgname[i].endswith('.mat'):
                img = loadmat(os.path.join(args.prob_dir, item, imgname[i]))['prob']
            else:
                img = imread(os.path.join(args.prob_dir, item, imgname[i]))
            out, new_dict, root = GetComFast(img, args.min_thres, args.min_cc)

            count = 1
            marker = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            for node in new_dict:
                if new_dict[node].is_leaf:
                    point = np.unravel_index(np.array(list(new_dict[node].max_point)), img.shape)
                    marker[point[0], point[1]] = count
                    count += 1

            imsave(os.path.join(args.ws_result_dir, item, imgname[i][:-4]+'.png'), marker)




if __name__ == "__main__":

      parser = ArgumentParser()
      parser.add_argument('--ws_result_dir', type=str, default="./App/")
      parser.add_argument('--prob_dir', type=str, default="")
      parser.add_argument('--min_thres', type=int, default=127)
      parser.add_argument('--min_cc', type=int, default=10)

      args = parser.parse_args()
      print(args)
      main(args)

