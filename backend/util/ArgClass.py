import argparse
import os
import pickle

import torch
from torch import nn

from pipnet.pipnet import PIPNet, get_network
from util.args import get_args, get_optimizer_nn
from util.func import get_device


class Args:
    def __init__(self, default=True, **kwargs):
        if default:
            self.args = get_args()
        else:
            self.args = argparse.ArgumentParser('Train a PIP-Net').parse_args()

        for key, value in kwargs.items():
            setattr(self.args, key, value)

        self.__device = get_device()

    def set_args(self, arg, value):
        setattr(self.args, arg, value)

    def get_arg(self, arg):
        return getattr(self.args, arg)

    def save_args(self, directory_path: str) -> None:
        """
        Save the arguments in the specified directory as
            - a text file called 'args.txt'
            - a pickle file called 'args.pickle'
        :param directory_path: The path to the directory where the arguments should be saved
        """
        # If the specified directory does not exist, create it
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        # Save the args in a text file
        with open(directory_path + '/args.txt', 'w') as f:
            for arg in vars(self.args):
                val = getattr(self.args, arg)
                if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                    val = f"'{val}'"
                f.write('{}: {}\n'.format(arg, val))

        # Pickle the args for possible reuse
        with open(directory_path + '/args.pickle', 'wb') as f:
            pickle.dump(self.args, f)

    def get_args(self):
        return self.args

    def load_args(self, directory_path: str):
        """
        Load the pickled arguments from the specified directory
        :param directory_path: The path to the directory from which the arguments should be loaded
        :return: the unpicked arguments
        """
        with open(directory_path + '/args.pickle', 'rb') as f:
            self.args = pickle.load(f)

    def get_optimizer_nn(self, net) -> torch.optim.Optimizer:
        return get_optimizer_nn(net, self.args)

    def get_patch_size(self):
        patch_size = 32
        skip = round((self.args.image_size - patch_size) / (self.args.wshape - 1))
        return patch_size, skip

    def load_net(self, state_dict_dir_net: str, num_classes: int = 2):

        feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(num_classes,
                                                                                                   self.args)
        net = PIPNet(num_classes=num_classes,
                     num_prototypes=num_prototypes,
                     feature_net=feature_net,
                     args=self.args,
                     add_on_layers=add_on_layers,
                     pool_layer=pool_layer,
                     classification_layer=classification_layer
                     )
        net = net.to(device=self.__device)
        net = nn.DataParallel(net)

        optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net,
                                                                                                                   self.args)
        with torch.no_grad():

            checkpoint = torch.load(state_dict_dir_net, map_location=self.__device)
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
                print("Classification layer initialized with mean",
                      torch.mean(net.module._classification.weight).item(),
                      flush=True)
                if self.args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            else:
                if 'optimizer_classifier_state_dict' in checkpoint.keys():
                    optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
        return net


