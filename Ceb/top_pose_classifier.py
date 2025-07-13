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

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, transformation):

        cls = ['0', '1']
        files_all, cls_all = [], []

        for i in range(len(cls)):
            f_temp = [os.path.join(path, cls[i], f) for f in os.listdir(os.path.join(path, cls[i])) if f.endswith('.png') or f.endswith('.tif')]
            cls_temp = [i]*len(f_temp)
            files_all.extend(f_temp)
            cls_all.extend(cls_temp)

        indexes = [i for i in range(len(files_all))]
        random.shuffle(indexes)

        self.files = [files_all[k] for k in indexes]
        self.cls = [cls_all[k] for k in indexes]
        self.trans = transformation

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        img = self.trans(img)

        return img, self.cls[index]

    def __len__(self):
        return len(self.files)

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(7*4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def train_model(model, dataloaders, criterion, optimizer, device, checkpoint_save_dir, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0)

    if len(dataloaders['val']) > 0:
        status = ['train', 'val']
    else:
        status = ['train']
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in status:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # loss = torchvision.ops.sigmoid_focal_loss(outputs, torch.nn.functional.one_hot(labels, num_classes=2).double(),
                        #                                           reduction='mean')

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'val':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if len(dataloaders['val']) == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f'epoch_{epoch:03d}.pth'))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


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


def main(args):
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
                # transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
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

    if args.phase == 'train':
        # Create training and validation datasets
        # image_datasets = {
        #     x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val']
        # }
        # image_datasets = datasets.ImageFolder(args.data_dir, data_transforms['train'])
        image_datasets = ImageDataset(args.data_dir, data_transforms['train'])
        dataloaders_dict = defaultdict()

        if args.val_dir is not "":
            val_datasets = ImageDataset(args.val_dir, data_transforms['val'])
            dataloaders_dict['train'] = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True,
                                                                    num_workers=4)
            dataloaders_dict['val'] = torch.utils.data.DataLoader(val_datasets, batch_size=args.batch_size, shuffle=True,
                                                                  num_workers=4)

        else:
            train_ds, val_ds = random_split(image_datasets, [1-args.val_ratio, args.val_ratio])
            dataloaders_dict['train'] = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                                                    num_workers=4)

            if len(val_ds) > 0:
                dataloaders_dict['val'] = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                                                    num_workers=4)
            else:
                dataloaders_dict['val'] = []

        params_to_update = model_ft.parameters()
        if args.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad is True:
                    params_to_update.append(param)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        # optimizer_ft = torch.optim.AdamW(params_to_update, lr=0.001, weight_decay=1e-4)

        # Setup the loss fxn
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss(alpha=torch.tensor([1.0, 1.0]).to("cuda:0"), gamma=2,reduction='mean')
        criterion = FocalLoss(gamma=2, reduction='mean')
        os.makedirs(args.checkpoint_save_dir, exist_ok=True)
        # Train and evaluate
        train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            optimizer_ft,
            device,
            args.checkpoint_save_dir,
            num_epochs=args.num_epochs,
            is_inception=(args.model_name == "inception"),
        )

    else:
        os.makedirs(os.path.join(args.result_dir), exist_ok=True)
        result_f = open(os.path.join(args.result_dir, 'negative_names.txt'), 'w')
        result_pos = open(os.path.join(args.result_dir, 'positive_names.txt'), 'w')
        model_ft.load_state_dict(torch.load(args.checkpoint))

        model_ft.eval()  # Set model to evaluate mode

        sub_dir = [f for f in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, f))]
        sub_dir.sort()
        if len(sub_dir) == 0:
            sub_dir = ['']

        print(sub_dir)
        correct_c, total_c = 0, 0
        pred_result = []
        for sub in sub_dir:
            # Iterate over data.
            sig = [f for f in os.listdir(os.path.join(args.test_dir, sub)) if f.endswith('.png') or '.jpg']
            pred_all = []
            for i in range(len(sig)):
                f = sig[i]
                try:
                    img = Image.open(os.path.join(args.test_dir, sub, f)).convert('RGB')
                except IOError:
                    continue
                inputs = data_transforms['val'](img)
                inputs = inputs.unsqueeze(0)
                inputs = inputs.to(device)

                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)

                pred = preds.cpu().numpy()[0]
                pred_all.append(pred)
                if pred == 0:
                    result_f.writelines(f[:-4]+'\n')
                elif pred == 1:
                    result_pos.writelines(f[:-4]+'\n')

            pred_all = np.array(pred_all)
            c_c = len(np.where(pred_all==int(sub))[0])
            t_c = len(pred_all)
            print('for {} class, {} out of {} are correct'.format(int(sub), c_c, t_c))
            if int(sub) >= 0:
                correct_c += c_c
                total_c += t_c

        print('accuracy: ', str(correct_c / total_c))
        if args.stat_file != '':
            with open(args.stat_file, 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                spamwriter.writerow([os.path.basename(args.checkpoint), str(correct_c / total_c)])

        if args.pred_file != '':
            pred_result = np.array(pred_result)
            np.savetxt(args.pred_file, pred_result, delimiter=",")



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
        '--checkpoint_save_dir', type=str, default='./checkpoints/top_best.pth', help='Path to save the checkpoint'
    )
    parser.add_argument('--test_dir', type=str, default="./img/", help='Test image directory')
    parser.add_argument('--val_dir', type=str, default="")
    parser.add_argument('--checkpoint', type=str, help='Load checkpoints for eval')
    parser.add_argument('--target', type=int, default=0, help='target positive class to save')
    parser.add_argument('--result_dir', type=str, default='./result/Top/', help='Directory to save positive class')
    parser.add_argument('--stat_file', type=str, default='')
    parser.add_argument('--feature_extract', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--pred_file', type=str, default='')

    args = parser.parse_args()
    main(args)
