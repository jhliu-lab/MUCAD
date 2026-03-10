import os
from enum import Enum

import PIL
import torch
import pandas as pd
from torchvision import transforms

_CLASSNAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VisADatatset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Vsia.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        test_val_split=0.5,
        csv_path="/home/xr/ljh/datasets/visa_sam_b/split_csv/1cls.csv",
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.csv_path = csv_path
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.test_val_split = test_val_split
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)
        self.transform_mean = [0.485, 0.456, 0.406]
        self.transform_std = [0.229, 0.224, 0.225]

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if (self.split == DatasetSplit.TEST or self.split == DatasetSplit.VAL) and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "normal"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        # load and handle csv file
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        if self.split == DatasetSplit.TRAIN:
            df = df[df['split'] == "train"]
        elif self.split == DatasetSplit.TEST:
            df = df[df['split'] == "test"]
        else:
            df = df[df['split'] == "test"]
        for classname in self.classnames_to_use:
            df_cls = df[df['object'] == classname]
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            for i in range(len(df_cls)):
                row_data = df_cls.iloc[i]
                if row_data['label'] not in imgpaths_per_class[classname].keys():
                    imgpaths_per_class[classname][row_data['label']] = [os.path.join(self.source, row_data['image'])]
                else:
                    imgpaths_per_class[classname][row_data['label']].append(os.path.join(self.source, row_data['image']))
                if row_data['label'] != 'normal':
                    if row_data['label'] not in maskpaths_per_class[classname].keys():
                        maskpaths_per_class[classname][row_data['label']] = [os.path.join(self.source, row_data['mask'])]
                    else:
                        maskpaths_per_class[classname][row_data['label']].append(os.path.join(self.source, row_data['mask']))
            if self.test_val_split < 1.0:
                for classname in sorted(imgpaths_per_class.keys()):
                    for label in sorted(imgpaths_per_class[classname].keys()):
                        images_num = len(imgpaths_per_class[classname][label])
                        test_val_split_idx = max(int(images_num * self.test_val_split),1)
                        if self.split == DatasetSplit.TRAIN:
                            #imgpaths_per_class[classname][label] = imgpaths_per_class[classname][label][:train_val_split_idx]
                            maskpaths_per_class[classname][label] = None
                        if self.split == DatasetSplit.VAL:
                            imgpaths_per_class[classname][label] = imgpaths_per_class[classname][label][test_val_split_idx-1:]
                            if label == "normal":
                                maskpaths_per_class[classname][label] = None
                            else:
                                maskpaths_per_class[classname][label] = maskpaths_per_class[classname][label][test_val_split_idx-1:]
                        if self.split == DatasetSplit.TEST:
                            imgpaths_per_class[classname][label] = imgpaths_per_class[classname][label][:test_val_split_idx]
                            if label == "normal":
                                maskpaths_per_class[classname][label] = None
                            else:
                                maskpaths_per_class[classname][label] = maskpaths_per_class[classname][label][:test_val_split_idx]
            else:
                for classname in sorted(imgpaths_per_class.keys()):
                    for label in sorted(imgpaths_per_class[classname].keys()):
                        if self.split == DatasetSplit.TRAIN or self.split == DatasetSplit.VAL:
                            maskpaths_per_class[classname][label] = None
                        if self.split == DatasetSplit.TEST and label == "normal":
                            maskpaths_per_class[classname][label] = None
 
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if (self.split == DatasetSplit.TEST or self.split == DatasetSplit.VAL) and anomaly != "normal":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
