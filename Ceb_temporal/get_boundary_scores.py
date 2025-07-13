""" top/pose image classifier

Refer to: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Training phase
input: a local directory contains all training data (train set and val set)
output: the best checkpoint (evaluated by the val set)

Example Usage:
python3 top_pose_classifier.py \
    --data_dir 'path/to/training_date/dir/' \
    --num_class 2 \
    --result_dir 'path/to/result/dir/' \
    --num_epoch 20

Inference phase
input: a local directory contains all downloaded images, checkpoint path
output: result_dir contains all top/pose images filtered by the classifier (predicted as top/pose class)

Example Usage:
python3 top_pose_classifier.py \
    --test_dir 'path/to/test_data/dir/' \
    --phase 'test' \
    --num_class 3 \
    --test_file "path/to/data/record.csv" \
    --checkpoint 'path/to/best_checkpoint.pth' \
    --result_dir 'path/to/result/dir/'

Training phase
Training images should be saved in the following format:
./Top/
    -train/
        -front/
            -img01.png
            -img02.png
            ...
        -other/
            -img01.png
            -img-2.png
            ...
    -val/
        -front/
            -img01.png
            -img02.png
            ...
        -other/
            -img01.png
            -img02.png

Test phase:
Test images are saved in the following format:
./img/
    -img01.png
    -img02.png
    ...
Positive class (--target) will be saved in --result_dir
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import copy
import os
import time

from collections import defaultdict
import csv
import cv2

import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision
from torchvision import datasets
from torchvision import models
from torchvision import transforms

from torch.utils.data import random_split
from typing import Optional, Sequence
from torch import Tensor
from torch.nn import functional as F

from skimage.io import imread, imsave
import pickle


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def initialize_model(model_name, num_classes, use_pretrained=True):

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        # Resnet18
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        # Alexnet
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        # VGG11_bn
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        # Squeezenet
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        # Densenet
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        # Inception v3
        # Be careful, expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "mnist":
        # Resnet18
        model_ft = ResNet9(3, num_classes)
        input_size = 112

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_ins_to_line(line_map):
   ins_to_line = defaultdict(int)

   for item in line_map:
      ins_to_line[line_map[item]] = item

   return ins_to_line

def get_all_images(test_dir):

    img_to_lines = defaultdict(list)

    sub_dir = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    sub_dir.sort()
    if len(sub_dir) == 0:
        sub_dir = ['']

    for sub in sub_dir:
        sig = [f for f in os.listdir(os.path.join(test_dir, sub)) if f.endswith('.png') or '.jpg']
        for f in sig:
            img_to_lines[f.split('_')[0]].append([os.path.join(test_dir, sub, f), int(sub)])

    return img_to_lines


def main(args):

    os.makedirs(os.path.dirname(args.reliable_file), exist_ok=True)

    model_ft, input_size = initialize_model(
        args.model_name, args.num_classes, use_pretrained=True
    )

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose(
            [
                transforms.RandomRotation(degrees = 360),
                transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                # transforms.ElasticTransform(alpha=120.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        'val': transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    print("Initializing Datasets and Dataloaders...")

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(args.checkpoint))

    model_ft.eval()  # Set model to evaluate mode

    result_f = open(args.reliable_file, 'w')

    m = nn.Softmax(dim=-1)
    img_dict = get_all_images(args.test_dir)
    count_total, count_correct = 0, 0
    incorrect_value = []
    for sub in img_dict:
        pm_img = imread(os.path.join(args.PM_dir, sub + '.png'))
        pm_img = cv2.cvtColor(pm_img, cv2.COLOR_GRAY2BGR)
        ws_line_img = imread(os.path.join(args.WS_line_dir, sub + '.png'))
        with open(os.path.join(args.WS_map_dir, sub + '.pickle'), 'rb') as hander:
            ws_map = pickle.load(hander)
        look_up = get_ins_to_line(ws_map)
        for item in img_dict[sub]:
            f, label = item[0], item[1]
            line_id = look_up[(int(os.path.basename(f).split('_')[1]), int(os.path.basename(f).split('_')[2].split('.')[0]))]
            y_cor, x_cor = ( np.max(np.where(ws_line_img==line_id)[0])+np.min(np.where(ws_line_img==line_id)[0]) )//2, \
                ( np.max(np.where(ws_line_img==line_id)[1])+np.min(np.where(ws_line_img==line_id)[1]) )//2

            pm_img[:,:,0][ws_line_img==line_id] = 255
            pm_img[:, :, 1][ws_line_img == line_id] = 0
            pm_img[:, :, 2][ws_line_img == line_id] = 0

            try:
                img = Image.open(f).convert('RGB')
            except IOError:
                continue
            inputs = data_transforms['val'](img)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)

            outputs = model_ft(inputs)
            outputs = m(outputs)

            value = outputs[0, 1].detach().cpu().numpy()

            result_f.writelines(os.path.basename(f)[:-4] + ",{:.3f}".format(1-value)+'\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="./Top", help='Training data directory')
    parser.add_argument(
        '--model_name',
        type=str,
        default="inception",
        help='Models to choose from [resnet, alexnet, \
                                                                vgg, squeezenet, densenet, inception]',
    )
    parser.add_argument('--phase', type=str, default="train", help='Phase to choose from [train, eval]')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs to train for')
    parser.add_argument(
        '--check_result_file', type=str, default='./checkpoints/top_best.pth', help='Path to save the checkpoint'
    )
    parser.add_argument('--test_dir', type=str, default="./img/", help='Test image directory')
    parser.add_argument('--checkpoint', type=str, help='Load checkpoints for eval')
    parser.add_argument('--target', type=int, default=0, help='target positive class to save')
    parser.add_argument('--stat_file', default='', type=str)
    parser.add_argument('--feature_extract', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--pred_file', type=str, default='')
    parser.add_argument('--WS_map_dir', type=str, default='')
    parser.add_argument('--PM_dir', type=str, default='')
    parser.add_argument('--WS_line_dir', type=str, default='')
    parser.add_argument('--reliable_file', type=str, default='')

    args = parser.parse_args()
    main(args)
