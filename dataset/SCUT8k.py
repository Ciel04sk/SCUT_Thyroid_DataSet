import torch.utils.data as data
import PIL.Image as Image
import os
import json
import numpy as np
import torch
from torchvision.transforms import transforms


class ThyroidDataset(data.Dataset):

    def __init__(self, mode, return_size=False, fold=0):
        self.mode = mode
        # ToDo (change the root)
        self.root = r"..."

        trainval = json.load(open(self.root + '/trainval'+'.json', 'r'))
        if mode == 'train':
            imgs = self.make_dataset(trainval['train'])
        elif mode == 'val':
            imgs = self.make_dataset(trainval['val'])

        self.imgs = imgs
        self.x_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.return_size = return_size

    def make_dataset(self, img_list):
        imgs = []
        for i in img_list:
            im = os.path.join(self.root, "image/{}.jpg".format(str(i).zfill(4)))
            mask = os.path.join(self.root, "mask/{}.png".format(str(i).zfill(4)))
            imgs.append((im, mask))
        return imgs

    def __getitem__(self, item):
        image_path, label_path = self.imgs[item]
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

        image = Image.open(image_path).convert('L')
        image = image.resize((224, 224), Image.BILINEAR)
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = self.x_transforms(image)

        mask = Image.open(label_path).convert('L')
        mask = mask.resize((224, 224), Image.BILINEAR)
        mask = np.array(mask, dtype=np.float32)
        mask = mask / 255.0
        mask = self.x_transforms(mask)

        sample = {'image': image, 'mask': mask}

        label_name = os.path.basename(label_path)
        sample['label_name'] = label_name
        sample["size"] = torch.tensor((224, 224))

        return sample

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    x = ThyroidDataset("val")
    print(x[0])
