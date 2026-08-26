import os
import logging

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import Image

import src.moco as moco

logger = logging.getLogger(__name__)


def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

def init_weights(model, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model



class UCMLCL(torchvision.datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, dual=False):
        super(UCMLCL, self).__init__(root, train, transform, target_transform)
        self.transform = transform
        self.dual = dual
        self.loader = default_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.dual:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        elif self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = sample

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.dual:
            return [sample1, sample2], target
        else:
            return sample, target

class CustomTensorDataset_CL(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.cls_num = 21
        self.targets = self.tensors[1]

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        x = np.array(x)
        x = Image.fromarray(np.uint8(x))

        if self.transform:
            x1 = self.transform[0](x)
            x2 = self.transform[1](x)
        return [x1,x2], y

    def __len__(self):
        return self.tensors[0].size(0)
    
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(torch.sum(self.targets.eq(i)).item())
        return cls_num_list

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def _split_dataset(train_dataset, class_num, client_num, non_iid_alpha, iid, batch_size):

    if iid == True:
        partition_proportions = np.full(shape=(class_num, client_num), fill_value=1/client_num)
    elif iid == False:
        partition_proportions = np.random.dirichlet(alpha=np.full(shape=client_num, fill_value=non_iid_alpha), size=class_num)
    
    client_train_datasets = _split_dataset_by_proportion(train_dataset, partition_proportions, class_num, client_num, iid, batch_size)

    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    transform_train = [transforms.Compose(augmentation_sim_cifar), transforms.Compose(augmentation_sim_cifar)]
    
    local_datasets = []
    for i in range(len(client_train_datasets)):
        trainsetlist = []
        trainlabellist = []
        client_train_datasets[i]
        for j in range(len(client_train_datasets[i])):
            d = np.array(client_train_datasets[i][j][0])
            data = np.resize(d,(256,256,3))
            trainsetlist.append(torch.tensor(data))
            trainlabellist.append(client_train_datasets[i][j][1])

        trainset = torch.stack(trainsetlist)
        trainlabel = torch.tensor(trainlabellist)
        local_datasets.append(CustomTensorDataset_CL(
                    (
                        trainset,
                        trainlabel.long()
                    ),transform=transform_train
                ))
    return local_datasets

def _split_dataset_by_proportion(dataset, partition_proportions, class_num, client_num, iid, batch_size):
    data_labels = dataset.targets
    class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
                  for y in range(class_num)]
    
    client_idcs = [[] for _ in range(client_num)]
    if iid == False:
        for _ in range(batch_size):# 每个客户端至少一个样本,至少让样本数和batch_size一样大，这样dataloader中的drop_last就不会让有的客户端没有数据
            for client_id in range(client_num):  
                class_id = np.random.randint(low=0, high=class_num)
                client_idcs[client_id].append(class_idcs[class_id].pop())
                
    for c, fracs in zip(class_idcs, partition_proportions):
        np.random.shuffle(c)
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i].extend(list(idcs))
    client_data_list = [[] for _ in range(client_num)]
    for client_id, client_data in zip(client_idcs, client_data_list):
        for id in client_id:
            client_data.append(dataset[id])
    
    client_datasets = []
    for client_data in client_data_list:
        np.random.shuffle(client_data)
        client_datasets.append(client_data)
    return client_datasets

def create_datasets_CL(data_path, dataset_name, num_clients, num_shards, iid, diri, alpha, batchsize):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    dataset_name = dataset_name.upper()

    augmentation_aug_cifar = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.2),
        #CIFAR10Policy(),    # add AutoAug
        transforms.ToTensor(),
        #Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    if dataset_name == "UCML":
        num_classes = 21
        ROOT_TRAIN = data_path + dataset_name + '/train'
        ROOT_TEST = data_path +  dataset_name + '/val'
        normalize = transforms.Normalize(mean=[0.48422759, 0.49005176, 0.45050278],
                                std=[0.17348298, 0.16352356, 0.15547497])

        train_transform = [transforms.Compose(augmentation_sim_cifar), transforms.Compose(augmentation_sim_cifar)]
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize])
        training_dataset = UCMLCL(
                root=ROOT_TRAIN,
                train=True,
                transform=None
            )
        test_dataset = torchvision.datasets.ImageFolder(ROOT_TEST, transform=test_transform)      
    elif dataset_name == "NWPU":
        num_classes = 45
        ROOT_TRAIN = data_path + dataset_name + '/train'
        ROOT_TEST = data_path +  dataset_name + '/val'
        normalize = transforms.Normalize(mean=[0.48422759, 0.49005176, 0.45050278],
                                std=[0.17348298, 0.16352356, 0.15547497])

        train_transform = [transforms.Compose(augmentation_sim_cifar), transforms.Compose(augmentation_sim_cifar)]
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize])
        training_dataset = UCMLCL(
                root=ROOT_TRAIN,
                train=True,
                transform=None
            )
        test_dataset = torchvision.datasets.ImageFolder(ROOT_TEST, transform=test_transform)
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    if dataset_name == 'UCML':
        data = np.ndarray((len(training_dataset),256,256,3))
        targets = []
        for i in range(len(training_dataset)):
            # data[i] = np.asarray(training_dataset[i][0]).transpose(1,2,0)
            d = np.asarray(training_dataset[i][0])
            data[i] = np.resize(d,(256,256,3))
            targets.append(training_dataset[i][1])
        training_dataset.data = data
        training_dataset.targets = targets
        num_categories = 21
    elif dataset_name == 'NWPU':
        data = np.ndarray((len(training_dataset),256,256,3))
        targets = []
        for i in range(len(training_dataset)):
            # data[i] = np.asarray(training_dataset[i][0]).transpose(1,2,0)
            d = np.asarray(training_dataset[i][0])
            data[i] = np.resize(d,(256,256,3))
            targets.append(training_dataset[i][1])
        training_dataset.data = data
        training_dataset.targets = targets
        num_categories = 45


    
    # split dataset according to iid flag
    if iid:
        local_datasets = _split_dataset(train_dataset=training_dataset,
                                           class_num=num_classes,
                                           client_num=num_clients,
                                           non_iid_alpha=alpha,
                                           iid=iid,
                                           batch_size=batchsize)
        return local_datasets, test_dataset
    else:
        if diri:
            # if dataset_name == 'UCML':
            #      local_datasets = _split_dataset_ucml(train_dataset=training_dataset,
            #                                class_num=num_classes,
            #                                client_num=num_clients,
            #                                non_iid_alpha=alpha,
            #                                iid=iid,
            #                                batch_size=batchsize)
            # else:
            local_datasets = _split_dataset(train_dataset=training_dataset,
                                        class_num=num_classes,
                                        client_num=num_clients,
                                        non_iid_alpha=alpha,
                                        iid=iid,
                                        batch_size=batchsize)
            return local_datasets, test_dataset
        else:
            # sort data by labels
            sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
            training_inputs = training_dataset.data[sorted_indices]
            training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

            # partition data into shards first
            shard_size = len(training_dataset) // num_shards 
            shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
            shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

            # sort the list to conveniently assign samples to each clients from at least two classes
            shard_inputs_sorted, shard_labels_sorted = [], []
            for i in range(num_shards // num_categories):
                for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                    shard_inputs_sorted.append(shard_inputs[i + j])
                    shard_labels_sorted.append(shard_labels[i + j])
                    
            # finalize local datasets by assigning shards to each client
            shards_per_clients = num_shards // num_clients


            # if dataset_name == 'UCML':
            #     local_datasets = [
            #     CustomTensorDatasetUCML(
            #         (
            #             torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
            #             torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
            #         ),
            #         transform=train_transform
            #     ) 
            #     for i in range(0, len(shard_inputs_sorted), shards_per_clients)
            #     ]
            # else:
            local_datasets = [
                CustomTensorDataset_CL(
                    (
                        torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                        torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                    ),
                    transform=train_transform
                ) 
                for i in range(0, len(shard_inputs_sorted), shards_per_clients)
            ]
    return local_datasets, test_dataset
