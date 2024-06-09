from .AFHQ_dataset import get_afhq_dataset
from .LSUN_dataset import get_lsun_dataset
from torch.utils.data import DataLoader
from .CUSTOME_dataset import get_custome_dataset
from .GALAXY_dataset import get_galaxy_dataset
from .RADIATION_dataset import get_radiation_dataset
from .CHURCH_dataset import get_church_dataset
from .CELEBA_dataset import get_celeba_dataset
from .BEDROOM_dataset import get_bedroom_dataset

def get_dataset(dataset_type, dataset_paths, config, target_class_num=None, custome=False):
    if dataset_type == 'AFHQ'and custome == False:
        train_dataset, test_dataset = get_afhq_dataset(dataset_paths['AFHQ'], config)
    elif dataset_type == "BEDROOM" and custome == True:
        train_dataset, test_dataset = get_lsun_dataset(dataset_paths['BEDROOM'], config)
    elif dataset_type == "CELEBA" and custome == True:
        train_dataset, test_dataset = get_celeba_dataset(dataset_paths['CELEBA'], config)
    elif dataset_type == "CHURCH" and custome == True:
        train_dataset, test_dataset = get_church_dataset(dataset_paths['CHURCH'], config)
    elif dataset_type == "CUSTOME" and custome == True:
        train_dataset, test_dataset = get_custome_dataset(dataset_paths['CUSTOME'], config)
    elif dataset_type == "GALAXY" and custome == True:
        train_dataset, test_dataset = get_galaxy_dataset(dataset_paths['GALAXY'], config)
    elif dataset_type == "RADIATION" and custome == True:
        train_dataset, test_dataset = get_radiation_dataset(dataset_paths['RADIATION'], config)
    else:
        raise ValueError

    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset, bs_train=1, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs_train,
        drop_last=True,
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        sampler=None,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {'train': train_loader, 'test': test_loader}


