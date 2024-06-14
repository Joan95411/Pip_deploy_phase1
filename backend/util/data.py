import numpy as np
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split
import PIL
from numpy.random import randint


def get_data(args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset == 'CUB-200-2011':
        return get_birds(True,
                         '/mnt/data/ffobbe/data/CUB_200_2011/dataset/train_crop',
                         '/mnt/data/ffobbe/data/CUB_200_2011/dataset/train',
                         '/mnt/data/ffobbe/data/CUB_200_2011/dataset/test_crop',
                         args.image_size, args.seed, args.validation_size,
                         '../../data/CUB_200_2011/dataset/train',
                         '../../data/CUB_200_2011/dataset/test_full')
    if args.dataset == 'pets':
        return get_pets(True,
                        '../../data/PETS/dataset37/train',
                        '../../data/PETS/dataset37/train',
                        '../../data/PETS/dataset37/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == 'partimagenet':
        return get_partimagenet(True,
                                '/groups/dso/meinauta/partimagenet/PartImageNet/all',
                                '/groups/dso/meinauta/partimagenet/PartImageNet/all',
                                None,
                                args.image_size,
                                args.seed,
                                args.validation_size)
    if args.dataset == 'CARS':
        return get_cars(True,
                        '../../data/cars/dataset/train',
                        '../../data/cars/dataset/train',
                        '../../data/cars/dataset/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == 'hips_rect':
        return get_mura(True,
                        '/mnt/data/ffobbe/data/hip/rectangle/trainval',
                        '/mnt/data/ffobbe/data/hip/rectangle/trainval',
                        '/mnt/data/ffobbe/data/hip/rectangle/trainval',
                        '/mnt/data/ffobbe/data/hip/rectangle/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == 'hips_big_rect':
        return get_mura(True,
                        '/mnt/data/ffobbe/data/hip/big_rectangle/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == 'hips_big_rect_07':
        return get_mura(True,
                        '/mnt/data/ffobbe/data/hip/big_rectangle_0.7/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle_0.7/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle_0.7/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle_0.7/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == 'hips_big_rect_1':
        return get_mura(True,
                        '/mnt/data/ffobbe/data/hip/big_rectangle_1/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle_1/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle_1/trainval',
                        '/mnt/data/ffobbe/data/hip/big_rectangle_1/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == 'hips_norm':
        return get_mura(True,
                        '/mnt/data/ffobbe/data/hip/normal/trainval',
                        '/mnt/data/ffobbe/data/hip/normal/trainval',
                        '/mnt/data/ffobbe/data/hip/normal/trainval',
                        '/mnt/data/ffobbe/data/hip/normal/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == 'ankle':
        return get_mura(True,
                        '/mnt/data/nproto/ankle/dataset/trainval',
                        '/mnt/data/nproto/ankle/dataset/trainval',
                        '/mnt/data/nproto/ankle/dataset/trainval',
                        '/mnt/data/nproto/ankle/dataset/test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    if args.dataset == "rsna":
        return get_mura(True,
                        '/mnt/data/ffobbe/data/rsna/sorted_train',
                        '/mnt/data/ffobbe/data/rsna/sorted_train',
                        '/mnt/data/ffobbe/data/rsna/sorted_train',
                        '/mnt/data/ffobbe/data/rsna/sorted_test',
                        args.image_size,
                        args.seed,
                        args.validation_size)
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')


def get_shortcut_data(args):
    if args.dataset == "hips_big_rect":
        return get_shortcut_mura(
            '/mnt/data/nproto/ankle/dataset/trainval',
            args.image_size,
            args.seed,
        )
    else:
        raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')


def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(
        args)

    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None

    num_workers = args.num_workers

    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor(
            [(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=to_shuffle,
                                              sampler=sampler,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=True
                                              )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                                              batch_size=pretrain_batchsize,
                                                              shuffle=to_shuffle,
                                                              sampler=sampler,
                                                              pin_memory=cuda,
                                                              num_workers=num_workers,
                                                              worker_init_fn=np.random.seed(args.seed),
                                                              drop_last=True
                                                              )

    else:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                                              batch_size=pretrain_batchsize,
                                                              shuffle=to_shuffle,
                                                              sampler=sampler,
                                                              pin_memory=cuda,
                                                              num_workers=num_workers,
                                                              worker_init_fn=np.random.seed(args.seed),
                                                              drop_last=True
                                                              )

    trainloader_normal = torch.utils.data.DataLoader(trainset_normal,
                                                     batch_size=args.batch_size,
                                                     shuffle=to_shuffle,
                                                     sampler=sampler,
                                                     pin_memory=cuda,
                                                     num_workers=num_workers,
                                                     worker_init_fn=np.random.seed(args.seed),
                                                     drop_last=True
                                                     )
    trainloader_normal_augment = torch.utils.data.DataLoader(trainset_normal_augment,
                                                             batch_size=args.batch_size,
                                                             shuffle=to_shuffle,
                                                             sampler=sampler,
                                                             pin_memory=cuda,
                                                             num_workers=num_workers,
                                                             worker_init_fn=np.random.seed(args.seed),
                                                             drop_last=True
                                                             )

    projectloader = torch.utils.data.DataLoader(projectset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=cuda,
                                                num_workers=num_workers,
                                                worker_init_fn=np.random.seed(args.seed),
                                                drop_last=False
                                                )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     pin_memory=cuda,
                                                     num_workers=num_workers,
                                                     worker_init_fn=np.random.seed(args.seed),
                                                     drop_last=False
                                                     )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes


"""

def get_dataloaders(args: argparse.Namespace, device):
    
    # Get data loaders
    
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(args)
    del trainset_normal
    del trainset_normal_augment
    trainset_sc_unlearning, trainset_sc_unlearning_tranformed = get_shortcut_data(args)

    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = args.num_workers
    
    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor([(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain 
    
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
                                        
    else:        
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    
    trainloader_sc_unlearning = torch.utils.data.DataLoader(trainset_sc_unlearning,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    trainloader_sc_unlearning_tranformed = torch.utils.data.DataLoader(trainset_sc_unlearning_tranformed,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    
    projectloader = torch.utils.data.DataLoader(projectset,
                                              batch_size = 1,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=False
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True, 
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, trainloader_pretraining, projectloader, testloader, test_projectloader, classes
"""


def create_datasets(transform1, transform2, transform_no_augment, num_channels: int, train_dir: str, project_dir: str,
                    test_dir: str, seed: int, validation_size: float, train_dir_pretrain=None, test_dir_projection=None,
                    transform1p=None):
    trainvalset = torchvision.datasets.ImageFolder(train_dir)
    classes = trainvalset.classes
    targets = trainvalset.targets
    indices = list(range(len(trainvalset)))

    train_indices = indices

    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError(
                "There is no test set directory, so validation size should be > 0 such that training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices, test_size=validation_size,
                                                       stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment),
                                          indices=test_indices)
        print("Samples in trainset:", len(indices), "of which", len(train_indices), "for training and ",
              len(test_indices), "for testing.", flush=True)
    else:
        testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)

    trainset = torch.utils.data.Subset(
        TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
    trainset_normal = torch.utils.data.Subset(
        torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(
        torchvision.datasets.ImageFolder(train_dir, transform=transforms.Compose([transform1, transform2])),
        indices=train_indices)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)

    if test_dir_projection is not None:
        testset_projection = torchvision.datasets.ImageFolder(test_dir_projection, transform=transform_no_augment)
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
        targets_pr = trainvalset_pr.targets
        indices_pr = list(range(len(trainvalset_pr)))
        train_indices_pr = indices_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=validation_size,
                                                                 stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(
            TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p, transform2=transform2),
            indices=train_indices_pr)
    else:
        trainset_pretraining = None

    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(
        targets)


def get_pets(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
             validation_size: float):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 48, img_size + 48)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.))
        ])

        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size)


def get_partimagenet(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
                     validation_size: float):
    # Validation size was set to 0.2, such that 80% of the data is used for training
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 48, img_size + 48)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size)


def get_birds(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
              validation_size: float, train_dir_pretrain=None, test_dir_projection=None):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 8, img_size + 8)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            # for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size, train_dir_pretrain, test_dir_projection, transform1p)


def get_cars(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
             validation_size: float):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
        ])

        transform2 = transforms.Compose([
            TrivialAugmentWideNoShapeWithColor(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])

    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size)


def get_mura(augment: bool, train_dir: str, train_supervised_dir: str, project_dir: str, test_dir: str, img_size: int,
             seed: int, validation_size: float, train_dir_pretrain=None):
    # print("\nTest set currently not used. Evaluates on validation set!\n", flush=True)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224 + 8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.Grayscale(3),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size)


def get_shortcut_mura(train_dir, img_size, seed):
    train_dir = '/mnt/data/ffobbe/data/hip/normal/trainval'
    # print("\nTest set currently not used. Evaluates on validation set!\n", flush=True)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform1 = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])

    def add_shortcut(tensor: PIL.Image.Image) -> PIL.Image.Image:
        to_image = transforms.ToPILImage()
        image = to_image(tensor)
        pixel_map = image.load()
        width, height = image.size
        # gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        # radius = width//20
        # if radius%2 == 0:
        #    radius += 1
        # gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        # gray = cv2.rectangle(gray, (width//4, 0), (width-width//4, height), (255, 255, 255), -1) # cover the informative part of the image, so the rectangle is not on the crotch
        # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        # print(input_image.shape)
        w = randint(width // 10, width // 9)  # orgi: (5, 50)
        h = randint(height // 10, height // 9)  # orgi: (5, 50)
        start_i = randint(0, width // 4)
        start_j = randint(0, width // 4)
        end_i = min(start_i + w, width)
        end_j = min(start_j + h, height)

        r = randint(200, 255)
        g = randint(200, 255)
        b = randint(200, 255)
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                # setting the pixel value.
                pixel_map[i, j] = (r, g, b)
        return image

    def add_shortcut2(tensor: PIL.Image.Image) -> PIL.Image.Image:
        image = tensor
        pixel_map = image.load()
        width, height = image.size
        # gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        # radius = width//20
        # if radius%2 == 0:
        #   radius += 1
        # gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        # gray = cv2.rectangle(gray, (width//4, 0), (width-width//4, height), (255, 255, 255), -1) # cover the informative part of the image, so the rectangle is not on the crotch
        # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        # print(input_image.shape)
        w = randint(width // 10, width // 9)  # orgi: (5, 50)
        h = randint(height // 10, height // 9)  # orgi: (5, 50)
        start_i = randint(0, width // 4)
        start_j = randint(0, width // 4)
        end_i = min(start_i + w, width)
        end_j = min(start_j + h, height)

        r = randint(200, 255)
        g = randint(200, 255)
        b = randint(200, 255)
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                # setting the pixel value.
                pixel_map[i, j] = (r, g, b)
        return image

    transform2 = transforms.Compose([
        transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
        transforms.Lambda(add_shortcut),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])

    transform1_augment = transforms.Compose([
        transforms.Resize(size=(img_size + 32, img_size + 32)),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224 + 8, scale=(0.95, 1.))
    ])
    transform2_augment = transforms.Compose([
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
        transforms.Lambda(add_shortcut2),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])

    trainvalset_pr = torchvision.datasets.ImageFolder(train_dir)
    targets_pr = trainvalset_pr.targets
    indices_pr = list(range(len(trainvalset_pr)))
    train_indices_pr = indices_pr
    test_dir = None
    if test_dir is None:
        subset_targets_pr = list(np.array(targets_pr)[indices_pr])
        train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=0.2, stratify=subset_targets_pr,
                                                             random_state=seed)

    trainset_shortcut_unlearn = torch.utils.data.Subset(
        TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1, transform2=transform2), indices=train_indices_pr)
    trainset_shortcut_unlearn_augment = torch.utils.data.Subset(
        TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1_augment, transform2=transform2_augment),
        indices=train_indices_pr)
    return trainset_shortcut_unlearn, trainset_shortcut_unlearn_augment  # TODO: change the second one to include other transformations


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
            # print("imgs: ", self.imgs[200:500], flush=True)
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)


# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {

            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
