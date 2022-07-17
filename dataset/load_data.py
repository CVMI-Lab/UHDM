import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile
import os

def create_dataset(
    args,
    data_path,
    mode='train',
):
    def _list_image_files_recursively(data_dir):
        file_list = []
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('gt.jpg'):
                    file_list.append(os.path.join(home, filename))
        file_list.sort()
        return file_list

    if args.DATA_TYPE == 'UHDM':
        uhdm_files = _list_image_files_recursively(data_dir=data_path)
        dataset = uhdm_data_loader(args, uhdm_files, mode=mode)
    elif args.DATA_TYPE == 'FHDMi':
        fhdmi_files = sorted([file for file in os.listdir(data_path + '/target') if file.endswith('.png')])
        dataset = fhdmi_data_loader(args, fhdmi_files, mode=mode)
    elif args.DATA_TYPE == 'TIP':
        tip_files = sorted([file for file in os.listdir(data_path + '/source') if file.endswith('.png')])
        dataset = tip_data_loader(args, tip_files, mode=mode)
    elif args.DATA_TYPE == 'AIM':
        if mode=='train':
            aim_files = sorted([file for file in os.listdir(data_path + '/moire') if file.endswith('.jpg')])
        else:
            aim_files = sorted([file for file in os.listdir(data_path + '/moire') if file.endswith('.png')])
        dataset = aim_data_loader(args, aim_files, mode=mode)
    else:
        print('Unrecognized data_type!')
        raise NotImplementedError
    data_loader = data.DataLoader(
        dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.WORKER, drop_last=True
    )

    return data_loader


class uhdm_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_tar = self.image_list[index]
        number = os.path.split(path_tar)[-1][0:4]
        path_src = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_moire.jpg'
        if self.mode == 'train':
            if self.loader == 'crop':
                if os.path.split(path_tar)[0][-5:-3] == 'mi':
                    w = 4624
                    h = 3472
                else:
                    w = 4032
                    h = 3024
                x = random.randint(0, w - self.args.CROP_SIZE)
                y = random.randint(0, h - self.args.CROP_SIZE)
                labels, moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_tar, path_src])

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number

        return data

    def __len__(self):
        return len(self.image_list)


class fhdmi_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in_gt = self.image_list[index]
        number = image_in_gt[4:9]
        image_in = 'src_' + number + '.png'
        if self.mode == 'train':
            path_tar = self.args.TRAIN_DATASET + '/target/' + image_in_gt
            path_src = self.args.TRAIN_DATASET + '/source/' + image_in
            if self.loader == 'crop':
                x = random.randint(0, 1920 - self.args.CROP_SIZE)
                y = random.randint(0, 1080 - self.args.CROP_SIZE)
                labels, moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_tar, path_src])

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            path_tar = self.args.TEST_DATASET + '/target/' + image_in_gt
            path_src = self.args.TEST_DATASET + '/source/' + image_in
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number

        return data

    def __len__(self):
        return len(self.image_list)


class aim_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in_gt = self.image_list[index]
        number = image_in_gt[0:6]
        image_in = number + '.jpg'
        if self.mode == 'train':
            path_tar = self.args.TRAIN_DATASET + '/clear/' + image_in_gt
            path_src = self.args.TRAIN_DATASET + '/moire/' + image_in
            if self.loader == 'crop':
                x = random.randint(0, 1024 - self.args.CROP_SIZE)
                y = random.randint(0, 1024 - self.args.CROP_SIZE)
                labels, moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_tar, path_src])

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            image_in = number + '.png'
            image_in_gt = number + '.png'
            path_tar = self.args.TEST_DATASET + '/clear/' + image_in_gt
            path_src = self.args.TEST_DATASET + '/moire/' + image_in
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number

        return data

    def __len__(self):
        return len(self.image_list)


class tip_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        t_list = [transforms.ToTensor()]
        self.composed_transform = transforms.Compose(t_list)

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in = self.image_list[index]
        image_in_gt = image_in[:-10] + 'target.png'
        number = image_in_gt[:-11]

        if self.mode == 'train':
            labels = self.default_loader(self.args.TRAIN_DATASET + '/target/' + image_in_gt)
            moire_imgs = self.default_loader(self.args.TRAIN_DATASET + '/source/' + image_in)

            w, h = labels.size
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            labels = labels.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            moire_imgs = moire_imgs.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))

            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)

        elif self.mode == 'test':
            labels = self.default_loader(self.args.TEST_DATASET + '/target/' + image_in_gt)
            moire_imgs = self.default_loader(self.args.TEST_DATASET + '/source/' + image_in)

            w, h = labels.size
            labels = labels.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            moire_imgs = moire_imgs.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))

            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)


        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        moire_imgs = self.composed_transform(moire_imgs)
        labels = self.composed_transform(labels)

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number

        return data

    def __len__(self):
        return len(self.image_list)



    
def default_loader(path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs
    
def crop_loader(crop_size, x, y, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.crop((x, y, x + crop_size, y + crop_size))
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs
    
def resize_loader(resize_size, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.resize((resize_size,resize_size),Image.BICUBIC)
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs

def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)

