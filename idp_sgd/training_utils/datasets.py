import copy
import math
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from numpy import ndarray
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from typing import Callable, Optional, Tuple, Any
from typing import Dict, List

TEST_BS = 256
BS = 256

PARAMS_MNIST = {
    'dataset_name': 'MNIST',
    'n_data': 60_000,
    'shape': (1, 28, 28),
    'mean': (0.1307,),
    'std': (0.3081,),
    'root': '',
}

PARAMS_CIFAR10 = {
    'dataset_name': 'CIFAR10',
    'n_data': 50_000,
    'shape': (3, 32, 32),
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.2023, 0.1994, 0.2010),
    'root': 'CIFAR10',
}


class SVHNAdapter(datasets.SVHN):
    """
    To align the interfaces between SVHN and CIFAR10. We changed the
    split parameter to train.
    """

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        if train is True:
            split = 'train'
        else:
            split = 'test'
        super(SVHNAdapter, self).__init__(root,
                                          split=split,
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)
        self.targets = self.labels


PARAMS_SVHN = {
    'dataset_name': 'SVHN',
    'n_data': 73_257,
    'shape': (3, 32, 32),
    'mean': (0.43768212, 0.44376972, 0.47280444),
    'std': (0.19803013, 0.20101563, 0.19703615),
    'root': 'SVHN',
}


def get_dataset_params(dataset_name: str, args, test: bool = False) -> Dict:
    """Get parameters of a specific dataset.

    Args:
        dataset_name (str): name of the dataset
        args: program parameters
        test (bool, optional): specifies if test data or training data is taken

    Returns:
        Parameters: dictionary that contains parameters of the dataset
    """
    individualize = 'None'
    if not test and args is not None and args.individualize == 'clipping':
        individualize = 'clipping'
    if dataset_name == 'MNIST':
        params = PARAMS_MNIST
        if individualize == 'clipping':
            # indexed dataset: returns index, data item, and label
            constructor = MNIST
        else:
            constructor = torchvision.datasets.MNIST
    elif dataset_name == 'CIFAR10':
        params = PARAMS_CIFAR10
        if individualize == 'clipping':
            # indexed dataset: returns index, data item, and label
            constructor = CIFAR10
        else:
            constructor = torchvision.datasets.CIFAR10
    elif dataset_name == 'SVHN':
        params = PARAMS_SVHN
        if individualize == 'clipping':
            # indexed dataset: returns index, data item, and label
            constructor = SVHN
        else:
            constructor = SVHNAdapter
    else:
        raise Exception(f"Unknown dataset {dataset_name}")
    params['constructor'] = constructor
    return params


def get_dataset(dataset_name: str, args, test: bool = False) -> Dataset:
    """Return a pytorch dataset by dataset name.

    Args:
        dataset_name (str): name of the dataset
        args: the program parameters
        test (bool, optional): specifies if test data or training data is taken

    Returns:
        Dataset: pytorch dataset
    """
    params = get_dataset_params(dataset_name=dataset_name, args=args, test=test)
    return params['constructor'](
        root=params['root'],
        train=not test,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=params['mean'],
                                             std=params['std'])
        ]),
    )


def get_duplications(budgets: ndarray, precision: float) -> ndarray:
    """Get duplications for each data point according to budgets (rounds to the
    nearest precision).

    Args:
        budgets (ndarray): _description_
        precision (float): _description_

    Returns:
        ndarray: _description_
    """
    p = int(1 / precision)
    budgets = np.around(np.array(budgets / np.min(budgets)) * p) / p
    tmp = np.copy(budgets)
    while np.any(np.abs(np.around(tmp, 2) - np.around(tmp)) > precision):
        tmp += budgets
    return np.around(tmp).astype(int)


def get_upsampled_data_loader(dataset: Dataset,
                              pp_budgets: ndarray,
                              batch_size: int,
                              n_workers: int,
                              precision: float = 0.1) -> DataLoader:
    """Return a data loader that up-samples the data points according to the
    per-point budgets.

    Args:
        dataset (Dataset): _description_
        pp_budgets (ndarray): _description_
        batch_size (int): _description_
        n_workers (int): _description_
        precision (float): _description_

    Returns:
        DataLoader: _description_
    """

    data = dataset.data.cpu().detach().numpy()
    targets = dataset.targets.cpu().detach().numpy()
    tmp_data = data.copy()
    tmp_labels = targets.copy()
    duplications = get_duplications(budgets=pp_budgets, precision=precision)
    for duplication in np.unique(duplications):
        # TODO: please document this code more.

        idx = np.nonzero(duplications == duplication)
        tmp_data = np.concatenate([
            tmp_data,
            np.concatenate([data[idx] for _ in range(duplication)])
        ])
        tmp_labels = np.concatenate([
            tmp_labels,
            np.concatenate([targets[idx] for _ in range(duplication)])
        ])
    dataset.data = torch.Tensor(tmp_data)
    dataset.targets = torch.Tensor(tmp_labels)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=n_workers,
                      shuffle=True,
                      pin_memory=True)


def get_mnist_data_loader(train: bool, bs: int, num_workers: int = 1):
    """Get MNIST data loader.

    Args:
        train (bool): _description_
        bs (int): _description_
        num_workers (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # TODO : design decision: do we train on larger mini-batches or also on
    # mini-batches of size 1 for better comparison?
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((PARAMS_MNIST['mean'],),
                                                 (PARAMS_MNIST['std'],))
            ]),
        ),
        batch_size=bs,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )


class MNIST(datasets.MNIST):
    """This is an MNIST dataset which returns the indices together with the
    data."""

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root,
                                    train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR10(datasets.CIFAR10):
    """This is an CIFAR10 dataset which returns the indices together with the
    data."""

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(CIFAR10, self).__init__(root,
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SVHN(datasets.SVHN):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        if train is True:
            split = 'train'
        else:
            split = 'test'
        super(SVHN, self).__init__(root,
                                   split=split,
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=download)

        self.targets = self.labels

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_indexed_dataset(dataset_name: str, args, test: bool = False) -> Dataset:
    """Return a pytorch dataset by dataset name. Extend its __get_item__()
        method to also return index.

    Args:
        dataset_name (str): name of the dataset
        args: program arguements
        test (bool, optional): specifies if test data or training data is taken

    Returns:
        Dataset: pytorch dataset
    """
    params = get_dataset_params(dataset_name=dataset_name, args=args)
    constructor = None
    if dataset_name == 'MNIST':
        constructor = MNIST
    elif dataset_name == 'CIFAR10':
        constructor = CIFAR10
    elif dataset_name == 'SVHN':
        constructor = SVHN
    return constructor(
        root=params['root'],
        train=not test,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((params['mean'],),
                                             (params['std'],))
        ]),
    )


def assign_budgets(n_data: int, budgets: List, ratios: List, args) -> ndarray:
    """Assign budgets for each point in the dataset according to the ratios.

    Args:
        n_data (int): number of data points to which budgets are to be allocated
            (60000)
        budgets (List): unique privacy budgets to be randomly allocated to data
            (eps = 1, eps = 3)
        ratios (List): relative ratios of points to have the different budgets
            (.7, .3)
        args: program arguments
    Returns:
        ndarray: allocated per-point budgets
            shape : N 
            each index returns the budget for that point
    """
    assert all(np.array(budgets) > 0), 'Privacy budgets must be positive!'
    assert len(
        np.unique(budgets)) == len(budgets), 'Budgets must be different!'
    assert math.fsum(ratios) == 1, f'Ratios sum up to {math.fsum(ratios)}, ' \
                                   f'but they must sum up to 1!'
    assert len(budgets) == len(ratios), 'Numbers of budgets and ratios must ' \
                                        'be equal!'
    indices = np.arange(n_data)
    pp_budgets = np.zeros(n_data)
    n_groups = len(ratios)
    if args.assign_budget == 'random':
        for group in range(n_groups - 1):
            idx = np.random.choice(a=indices,
                                   size=int(round(n_data * ratios[group])),
                                   replace=False)
            pp_budgets[idx] = budgets[group]
            indices = np.setdiff1d(indices, idx)
        pp_budgets[pp_budgets == 0] = budgets[-1]
    elif args.assign_budget == 'even':
        size = len(pp_budgets) // n_groups
        for i in range(n_groups - 1):
            pp_budgets[i * size: (i + 1) * size] = budgets[i]
        pp_budgets[(n_groups - 1) * size:] = budgets[-1]
    elif args.assign_budget == 'per-class':
        assert all(np.sort(np.unique(args.class_budgets)) == np.sort(
            np.unique(budgets))), 'All class_budgets must be in budgets!'
        pp_budgets = assign_budgets_per_class(
            pp_labels=args.pp_labels,
            class_budgets=args.class_budgets)
    return pp_budgets


def assign_budgets_per_class(pp_labels: ndarray, class_budgets: List[float]) \
        -> ndarray:
    assert all(np.array(class_budgets) > 0), 'Privacy budgets must be positive!'
    classes = np.sort(np.unique(pp_labels))
    n_classes = len(classes)
    assert len(class_budgets) == n_classes, 'Each class requires one budget!'
    n_data = len(pp_labels)
    pp_budgets = np.zeros(n_data)
    for c in range(n_classes):
        idx = pp_labels == c
        pp_budgets[idx] = class_budgets[c]
    return pp_budgets


def get_pp_sample_rates(pp_budgets: ndarray,
                        relative_sample_rates: List[float],
                        batch_size: int) -> ndarray:
    """Compute per-point sample-rates according to the given relative
    sample-rates and ratios such that the expected batch-size equals the given
    batch-size.
    
    pp_budgets: ndarray: length N (dataset size)

    """
    assert any(np.array(relative_sample_rates) > 0), \
        'All relative sample rates equal 0!'
    unique_budgets = np.unique(pp_budgets)
    assert len(unique_budgets) == len(relative_sample_rates), \
        f'The length of unique budgets {len(unique_budgets)} must equal' \
        f'the length of relative_sample_rates {len(relative_sample_rates)}' \
        f'!'
    pp_sample_rates = np.zeros(len(pp_budgets))
    for i, sample_rate in enumerate(relative_sample_rates):
        pp_sample_rates[pp_budgets == unique_budgets[i]] = sample_rate
    pp_sample_rates *= batch_size / sum(pp_sample_rates)
    unique_sample_rates = np.unique(pp_sample_rates)
    idx = np.logical_or(unique_sample_rates < 0, np.unique(pp_sample_rates) > 1)
    assert all(pp_sample_rates >= 0) and all(pp_sample_rates <= 1), \
        f'There is a sample rate that is <0 or >1: ' \
        f'{unique_sample_rates[idx]}!'
    assert math.isclose(batch_size, sum(pp_sample_rates), abs_tol=1e-3), \
        f'The sum of pp_sample_rates {sum(pp_sample_rates)} must equal ' \
        f'batch_size {batch_size}!'
    return pp_sample_rates


def partition_dataset(dataset: Dataset, pp_budgets: ndarray) -> List[Dataset]:
    """Split given dataset into partitions according to per-point budgets.

        Args:
            dataset (Dataset): pytorch dataset to be split
            pp_budgets (ndarray): randomly allocated per-point budgets

        Returns:
            List[Dataset]: pytorch datasets that constitute partitions
        """
    partitions = [copy.copy(dataset) for _ in np.unique(pp_budgets)]
    for i, budget in enumerate(np.unique(pp_budgets)):
        partitions[i].data = partitions[i].data[pp_budgets == budget]
        partitions[i].targets = partitions[i].targets[pp_budgets == budget]
    return partitions


def get_grouped_loaders(dataset: Dataset, budgets: List[float],
                        ratios: List[float], batch_size: int, n_workers: int) \
        -> List[DataLoader]:
    """Create pytorch data loaders per privacy group.

    Args:
        dataset (Dataset): pytorch dataset to be split
        budgets (List): unique privacy budgets to be randomly allocated to data
        ratios (List): relative ratios of points to have the different budgets
        batch_size (int): number of data points used per gradient descent step
        n_workers (int): number of cpu cores to help passing data to the gpu

    Returns:
        List: pytorch data loaders per privacy group
    """
    pp_budgets = assign_budgets(n_data=len(dataset),
                                budgets=budgets,
                                ratios=ratios)
    partitions = partition_dataset(dataset=dataset, pp_budgets=pp_budgets)
    grouped_loaders = [
        DataLoader(
            partition,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            pin_memory=True,
        ) for partition in partitions
    ]
    return grouped_loaders
