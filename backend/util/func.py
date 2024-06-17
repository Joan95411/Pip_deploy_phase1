from backend.pipnet.pipnet import PIPNet, get_network
from backend.util.args import get_args, get_optimizer_nn
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
import shutil
import random
import re
import os
import torch


def get_patch_size(args):
    patchsize = 32
    skip = round((args.image_size - patchsize) / (args.wshape - 1))
    return patchsize, skip


def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))


# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=3662215#gistcomment-3662215
def topk_accuracy(output, target, topk=[1, ]):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        topk2 = [x for x in topk if x <= output.shape[1]]  # ensures that k is not larger than number of classes
        maxk = max(topk2)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            if k in topk2:
                correct_k = correct[:k].reshape(-1).float()
                res.append(correct_k)
            else:
                res.append(torch.zeros_like(target))
        return res


# functions to set up network outside of run

def load_net(state_dict_dir_net: str, num_classes: int = 2):
    args = get_args()

    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(num_classes, args)
    net = PIPNet(num_classes=num_classes,
                 num_prototypes=num_prototypes,
                 feature_net=feature_net,
                 args=args,
                 add_on_layers=add_on_layers,
                 pool_layer=pool_layer,
                 classification_layer=classification_layer
                 )
    net = net.to(device=device)
    net = nn.DataParallel(net)

    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net,
                                                                                                               args)
    with torch.no_grad():

        checkpoint = torch.load(state_dict_dir_net, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
        print("Pretrained network loaded", flush=True)
        classes = ["frac", "rect_no_frac"]
        if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(
                net.module._classification.weight).item() < 3.0 and torch.count_nonzero(
            torch.relu(net.module._classification.weight - 1e-5)).float().item() > 0.8 * (num_prototypes * len(
            classes)):  # assume that the linear classification layer is not yet trained (e.g. when loading a pretrained model only)
            print("We assume that the classification layer is not yet trained. We re-initialize it...", flush=True)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False
            print("Multiplier initialized with", torch.mean(net.module._multiplier).item(), flush=True)
            print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(),
                  flush=True)
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
        else:
            if 'optimizer_classifier_state_dict' in checkpoint.keys():
                optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
    return net


def get_device():
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    print("Device used: ", device)
    return device


def get_project_loader(vis_dir, args, number=None, norm_dir=None):
    img_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])
    if number:
        pattern = r".*/"
        match = re.search(pattern, vis_dir)
        vis_dir_new = f"{match.group(0)}/vis_dir"
        print(vis_dir_new)
        if not os.path.exists(vis_dir_new):
            os.makedirs(vis_dir_new)
        for root, dirs, files in os.walk(vis_dir):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.path.exists(f"{vis_dir_new}/{dir}"):
                    os.makedirs(f"{vis_dir_new}/{dir}")
                image_files = [file for file in os.listdir(dir_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
                random.seed(42)
                random.shuffle(image_files)
                selected_files = image_files[:number]

                for file in selected_files:
                    source_path = os.path.join(dir_path, file)
                    destination_path = os.path.join(f"{vis_dir_new}/{dir}", file)
                    shutil.copy(source_path, destination_path)
                    print(f"Moved file: {file}")

        vis_dir = vis_dir_new

    projectset = torchvision.datasets.ImageFolder(vis_dir, transform=transform_no_augment)
    projectloader = torch.utils.data.DataLoader(projectset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=1,
                                                worker_init_fn=np.random.seed(args.seed),
                                                drop_last=False
                                                )
    return projectloader


def get_comparison_project_loader(vis_dir, norm_dir, args, number=None):
    img_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])
    if number:
        pattern = r".*/"
        match = re.search(pattern, vis_dir)
        match2 = re.search(pattern, norm_dir)
        vis_dir_new = f"{match.group(0)}/vis_dir"
        print(vis_dir_new)
        norm_dir_new = f"{match2.group(0)}/vis_dir"
        print(norm_dir_new)
        if not os.path.exists(vis_dir_new):
            os.makedirs(vis_dir_new)
        if not os.path.exists(norm_dir_new):
            os.makedirs(norm_dir_new)
        for root, dirs, files in os.walk(vis_dir):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.path.exists(f"{vis_dir_new}/{dir}"):
                    os.makedirs(f"{vis_dir_new}/{dir}")
                if not os.path.exists(f"{norm_dir_new}/{dir}"):
                    os.makedirs(f"{norm_dir_new}/{dir}")
                image_files = [file for file in os.listdir(dir_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
                random.seed(42)
                random.shuffle(image_files)
                selected_files = image_files[:number]

                for file in selected_files:
                    source_path = os.path.join(dir_path, file)
                    destination_path = os.path.join(f"{vis_dir_new}/{dir}", file)
                    shutil.copy(source_path, destination_path)
                    print(f"Moved file: {file}")
                    pattern = r"^(.*?)_"
                    match = re.search(pattern, file)
                    if match:
                        short_file = match.group(1)
                        short_file = short_file + ".png"
                    else:
                        short_file = file
                    print(short_file)
                    for folder_name in os.listdir(norm_dir):
                        try:
                            source_path = os.path.join(f"{norm_dir}/{folder_name}", short_file)
                            destination_path = os.path.join(f"{norm_dir_new}/{dir}", short_file)
                            shutil.copy(source_path, destination_path)
                            print(f"Moved file: {short_file}")
                        except:
                            print("not the right folder")

                            source_path = os.path.join(f"{norm_dir}/{folder_name}", short_file)
                            destination_path = os.path.join(f"{norm_dir_new}/{dir}", short_file)
                            print(source_path, destination_path)

        vis_dir = vis_dir_new
        norm_dir = norm_dir_new

    projectset = torchvision.datasets.ImageFolder(vis_dir, transform=transform_no_augment)
    projectloader = torch.utils.data.DataLoader(projectset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=1,
                                                worker_init_fn=np.random.seed(args.seed),
                                                drop_last=False
                                                )

    normset = torchvision.datasets.ImageFolder(norm_dir, transform=transform_no_augment)
    normloader = torch.utils.data.DataLoader(normset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    return projectloader, normloader
