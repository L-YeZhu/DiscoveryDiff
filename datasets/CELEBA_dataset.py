from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os
import glob


class CELEBA_dataset(Dataset):
    def __init__(self, image_root, transform=None, mode='train', img_size=256):
        super().__init__()
        self.transform = transform
        self.img_size = img_size
        self.image_paths = glob.glob(os.path.join(image_root,"*"))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        x = x.resize((self.img_size, self.img_size))


        if self.transform is not None:
            x = self.transform(x)



        return x

    def __len__(self):
        return len(self.image_paths)



def get_celeba_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = CELEBA_dataset(data_root, transform=train_transform, mode='train', 
                                 img_size=config.data.image_size)
    test_dataset = CELEBA_dataset(data_root, transform=test_transform, mode='val',
                                img_size=config.data.image_size)

    return train_dataset, test_dataset
