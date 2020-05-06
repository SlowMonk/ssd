# https://mf1024.github.io/2019/06/22/Create-Pytorch-Datasets-and-Dataloaders/
# How to create and use custom pytorch Dataset from the Imagenet

from torch.utils.data import DataLoader

#from dataset import *
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import time

import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

IMG_SIZE = (128,128)
BATCH_SIZE=32



class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):

        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        #sample = {'img': img, 'my_annotation': my_annotation}
        return img, my_annotation
        #return sample



    def __len__(self):
        return len(self.ids)



class cocoDatasetTemp(object):
    def __init__(self):
        self.coco={

        'path': '/media/jake/mark-4tb3/input/datasets/coco',
        'train': '/media/jake/mark-4tb3/input/datasets/coco/train2017',
        'test': '/media/jake/mark-4tb3/input/datasets/coco/test2017',
        'path2json': '/media/jake/mark-4tb3/input/datasets/coco/instances_train2017.json',
        'save_images' : '/media/jake/mark-4tb3/input/datasets/coco/images/'
        }

    def get_train(self):
        train = dset.CocoDetection(root=self.coco['train'], annFile=self.coco['path2json'])
        return train

    def draw_box(self,train,num):
        img,target = train[num]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

        img_org = img.copy()
        blue_color = (255, 0, 0)
        img = np.array(img)
        for i in range(len(target)):
            bbox = target[i]['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x, y, w, h = int(x), int(y), int(w), int(h)
            # img_bbox=cv2.rectangle(train_image, (int(x),int(y)), (int(x)+int(w),int(y)+int(h)), (0,255,0), 10)
            img_bbox = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        im = Image.fromarray(img_bbox)
        im.save("./images/your_file.jpeg")
        ax1.imshow(img_org)
        ax2.imshow(img)


class ImageNetDataset(Dataset):
    #  데이터셋의 전처리를 해주는 부분
    def __init__(self, data_path, is_train, train_split = 0.9, random_seed = 42, target_transform = None, num_classes = None):
        super(ImageNetDataset, self).__init__()
        self.data_path = data_path

        self.is_classes_limited = False

        if num_classes != None:
            self.is_classes_limited = True
            self.num_classes = num_classes

        self.classes = []
        class_idx = 0
        for class_name in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path,class_name)):
                continue
            self.classes.append(
               dict(
                   class_idx = class_idx,
                   class_name = class_name,
               ))
            class_idx += 1

            if self.is_classes_limited:
                if class_idx == self.num_classes:
                    break

        if not self.is_classes_limited:
            self.num_classes = len(self.classes)

        self.image_list = []
        for cls in self.classes:
            class_path = os.path.join(data_path, cls['class_name'])
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.image_list.append(dict(
                    cls = cls,
                    image_path = image_path,
                    image_name = image_name,
                ))

        self.img_idxes = np.arange(0,len(self.image_list))

        np.random.seed(random_seed)
        np.random.shuffle(self.img_idxes)

        last_train_sample = int(len(self.img_idxes) * train_split)
        if is_train:
            self.img_idxes = self.img_idxes[:last_train_sample]
        else:
            self.img_idxes = self.img_idxes[last_train_sample:]
    # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return len(self.img_idxes)
    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):

        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]

        img = Image.open(img_info['image_path'])

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)

        tr = transforms.ToTensor()
        img1 = tr(img)

        width, height = img.size
        if min(width, height)>IMG_SIZE[0] * 1.5:
            tr = transforms.Resize(int(IMG_SIZE[0] * 1.5))
            img = tr(img)

        width, height = img.size
        if min(width, height)<IMG_SIZE[0]:
            tr = transforms.Resize(IMG_SIZE)
            img = tr(img)

        tr = transforms.RandomCrop(IMG_SIZE)
        img = tr(img)

        tr = transforms.ToTensor()
        img = tr(img)

        if (img.shape[0] != 3):
            img = img[0:3]

        return dict(image = img, cls = img_info['cls']['class_idx'], class_name = img_info['cls']['class_name'])

    def get_number_of_classes(self):
        return self.num_classes

    def get_number_of_samples(self):
        return self.__len__()

    def get_class_names(self):
        return [cls['class_name'] for cls in self.classes]

    def get_class_name(self, class_idx):
        return self.classes[class_idx]['class_name']
