from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os
import glob
import numpy as np
import torch

class CUSTOME_dataset(Dataset):
    def __init__(self, image_root, transform=None, mode='train', img_size=256):
        super().__init__()
        print("image_root:", os.path.join(image_root, mode))
        self.transform = None
        self.img_size = img_size
        self.data = torch.load("/n/fs/yz-diff/UnseenDiffusion/data/iddpm_dog_ood_galaxy_t1000.pt")
        # print("check latent data size:", self.data.size())
        # self.image_paths = self.image_paths_pos + self.image_paths_neg
        # print("check image_paths:", self.image_paths_pos, len(self.image_paths_pos))
        # print("check image_paths:", self.image_paths_neg, len(self.image_paths_pos))
        # print("check image_paths:", self.image_paths, len(self.image_paths))
        # exit()

    def __getitem__(self, index):
        # image_path = self.image_paths[index]
        x = self.data[index]

        return x

    def __len__(self):
        return self.data.size()[0]


################################################################################

def get_custome_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = CUSTOME_dataset(data_root, transform=None, mode='train', 
                                 img_size=config.data.image_size)
    test_dataset = CUSTOME_dataset(data_root, transform=None, mode='val',
                                img_size=config.data.image_size)

    return train_dataset, test_dataset


################################################################################

# def get_celeba_dataset(data_root, config):
#     train_transform = tfs.Compose([tfs.ToTensor(),
#                                    tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
#                                                  inplace=True)])

#     test_transform = tfs.Compose([tfs.ToTensor(),
#                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
#                                                 inplace=True)])

#     train_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_train'),
#                                            train_transform, config.data.image_size)
#     test_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_test'),
#                                           test_transform, config.data.image_size)


#     return train_dataset, test_dataset



def file_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip() for f in files]
    return files