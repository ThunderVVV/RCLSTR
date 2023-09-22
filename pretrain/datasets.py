import sys
import six
import os

import PIL
import lmdb
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
import numpy as np


class light_augment(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, opt):
        self.opt = opt

        # light augment
        self.Augment = iaa.Sequential([iaa.SomeOf((1, 5),
                [
                iaa.LinearContrast((0.5, 1.0)),
                iaa.GaussianBlur((0.5,1.5)),
                iaa.Crop(percent=((0, 0.4), (0, 0), (0, 0.4), (0, 0)), keep_size=True),
                iaa.Crop(percent=((0, 0), (0, 0.02), (0, 0), (0, 0.02)), keep_size=True),
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                iaa.PiecewiseAffine(scale=(0.02, 0.03), mode='edge'),
                iaa.PerspectiveTransform(scale=(0.01,0.02))                
                ], random_order=True)]
        )
        self.toTensor = transforms.ToTensor()
        print("Use light_augment", self.Augment)

    def __call__(self, x):
        x = x.resize((self.opt.imgW, self.opt.imgH), PIL.Image.BICUBIC)
        x = np.array(x)
        q = self.Augment.augment_image(x)
        q = self.toTensor(q)
        k = self.Augment.augment_image(x)
        k = self.toTensor(k)
        q.sub_(0.5).div_(0.5)
        k.sub_(0.5).div_(0.5)

        return [q, k]


class ImgDataset(Dataset):
    def __init__(self, root, opt, transforms="None"):

        self.root = root
        self.opt = opt
        if opt.light_aug:
            self.light_augment = light_augment(opt)
        else:
            self.transforms = transforms
        self.imgs = sorted(os.listdir(self.root))
        self.nSamples = len(self.imgs)
        

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = PIL.Image.open(os.path.join(self.root, self.imgs[index]))
        label = "[dummy_label]"
        if self.opt.light_aug:
            img = self.light_augment(img)
            # print(img)
        else:
            img = self.transforms(img)

        return (img, label)


class LmdbDataset(Dataset):
    def __init__(self, root, opt, transforms="None"):

        self.root = root
        self.opt = opt
        if opt.light_aug:
            self.light_augment = light_augment(opt)
        else:
            self.transforms = transforms
        env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = txn.get(label_key).decode("utf-8")

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        env = lmdb.open(
            self.root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            print("cannot open lmdb from %s" % (self.root))
            sys.exit(0)
            
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with env.begin(write=False) as txn:
            # label_key = "label-%09d".encode() % index
            # label = txn.get(label_key).decode("utf-8")
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"
        if self.opt.light_aug:
            img = self.light_augment(img)
            # print(img)
        else:
            img = self.transforms(img)
        label = "[dummy_label]"

        return (img, label)