from argparse import ArgumentParser
import numpy as np
import os
import cvxpy as cp
import pickle
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from collections import defaultdict
from skimage.morphology import thin, remove_small_objects

def pad_extra(img, extra_img):
    all_idx = np.unique(extra_img)
    all_idx = all_idx[all_idx>0]

    ovearlap = np.unique(extra_img[img > 0])
    ovearlap = ovearlap[ovearlap > 0]

    left_idx = set(all_idx).difference(set(ovearlap))

    count = np.amax(img) + 1

    for idx in left_idx:
        img[extra_img == idx] = count
        count += 1

    return img

def main(args):

    sub_file = [f for f in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, f))]
    sub_file.sort()
    if len(sub_file) == 0:
        sub_file = ['']
    print(sub_file)

    for sub in sub_file:
        file_names = [f for f in os.listdir(os.path.join(args.img_dir, sub)) if f.endswith('.png') or f.endswith('.tif')]
        file_names.sort()

        watershed_names = [f for f in os.listdir(os.path.join(args.watershed_dir, sub)) if
                      f.endswith('.png') or f.endswith('.tif')]
        watershed_names.sort()

        pm_names = [f for f in os.listdir(os.path.join(args.watershed_dir, sub)) if
                           f.endswith('.png') or f.endswith('.tif')]
        pm_names.sort()

        if args.extra_dir != "":
            extra_names = [f for f in os.listdir(os.path.join(args.extra_dir, sub)) if
                          f.endswith('.png') or f.endswith('.tif')]
            extra_names.sort()

        os.makedirs(os.path.join(args.app_result_dir, sub), exist_ok=True)
        for k, f in enumerate(file_names):
            img = np.uint16(imread(os.path.join(args.img_dir, sub, f)))

            if args.extra_dir != "":
                extra_img = imread(os.path.join(args.extra_dir, sub, extra_names[k]))
                img = pad_extra(img, extra_img)
            watershed = imread(os.path.join(args.watershed_dir, sub, watershed_names[k]))
            PM = imread(os.path.join(args.PM_dir, sub, pm_names[k]))
            PM = label(PM > 127)
            idxes = np.unique(PM)
            idxes = idxes[idxes > 0]
            watershed[img>0] = 0

            max_value = np.amax(img) + 1
            for idx in idxes:
                overlap = np.unique(watershed[PM==idx])
                overlap = overlap[overlap > 0]
                for k in overlap:
                    img[watershed==k] = max_value
                max_value += 1

            imsave(os.path.join(args.app_result_dir, sub, f), np.uint16(img))


if __name__ == "__main__":

      parser = ArgumentParser()
      parser.add_argument('--PM_dir', type=str, default="./data/")
      parser.add_argument('--img_dir', type=str, default="./data/")
      parser.add_argument('--extra_dir', type=str, default="")
      parser.add_argument('--watershed_dir', type=str, default="./data/")
      parser.add_argument('--min_region', type=int, default=30)
      parser.add_argument('--app_result_dir', type=str, default="./App/")

      args = parser.parse_args()
      main(args)