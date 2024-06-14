import os
from typing import Tuple, List

import torch
from PIL import Image
from torchvision import transforms

from InferenceResult import HipFractureImage, HipFracturePrototype, HipFractureEnum
from pipnet.pipnet import PIPNet
from util.ArgClass import Args
from util.func import get_device
from util.vis_pipnet import get_img_coordinates


class PIPNetInference:
    """
    Class for performing inference with a PIPNet model.
    """

    def __init__(self, net: PIPNet = None, args: Args = None):
        """
        :param net: The model to perform inference with.
        """
        self.__device = get_device()

        if not args:
            self.args = Args()
        else:
            self.args = args

        self.net = net
        if self.net:
            self.net.to(self.__device)
            self.net.eval()

    def load_model(self, model_dir: str) -> None:
        if self.net:
            print("Overwriting existing model.")

        self.net = self.args.load_net(model_dir)
        self.net.to(self.__device)
        self.net.eval()

    def get_transformation_arguments(self, mean: tuple = None, std: tuple = None) -> transforms.Compose:
        if not mean:
            mean = (0.485, 0.456, 0.406)
        if not std:
            std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=mean, std=std)
        return transforms.Compose([
            transforms.Resize(size=(self.args.get_arg('image_size'), self.args.get_arg('image_size'))),
            transforms.ToTensor(),
            normalize])

    def images_inference(self, directory: str):
        """
        Performs inference on a directory of images.
        """
        pass

    def dataloader_inference(self,
                             dataloader: torch.utils.data.DataLoader) -> \
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Performs inference on a dataset.
        :param dataloader: The dataset to perform inference on.
        :return: The softmax, pooled, and out tensors.
        """
        softmax_list, pooled_list, out_list = [], [], []
        for _, (xs, _) in enumerate(dataloader):  # shuffle is false so should lead to same order as in imgs
            with torch.no_grad():
                softmax_layer, pooled_layer, output_layer = self.single_tensor_inference(xs)
                # softmax has shape (bs, num_prototypes, W, H),
                # pooled has shape (bs, num_prototypes)
                # out has shape (bs, )

                self.args.set_args("wshape", softmax_layer.shape[-1])

                softmax_list.append(softmax_layer)
                pooled_list.append(pooled_layer)
                out_list.append(output_layer)

        return softmax_list, pooled_list, out_list

    def single_tensor_inference(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs inference on a single image.
        :param xs: The image to perform inference on.
        :return: The softmax, pooled, and out tensors.
        """
        xs = xs.to(self.__device)
        with torch.no_grad():
            softmax_layer, pooled_layer, output_layer = self.net(xs, inference=True)
            # softmax has shape (bs, num_prototypes, W, H),
            # pooled has shape (bs, num_prototypes)
            # out has shape (bs, )

            self.args.set_args("wshape", softmax_layer.shape[-1])
            return softmax_layer, pooled_layer, output_layer

    def single_image_inference(self, img: Image.Image, threshold=0.1) -> HipFractureImage:
        """
        Performs inference on a single image.
        :param img: The image to perform inference on.
        :param threshold: The weight threshold, below which patches are not considered.
        :return: The image with the predicted class and the images of the patches that were used to make the prediction.
        """
        transform_arg = self.get_transformation_arguments()
        image = transform_arg(img.convert('RGB'))
        softmax_layer, pooled_layer, output_layer = self.single_tensor_inference(image.unsqueeze_(0))
        return self.get_prediction_images(softmax_layer, pooled_layer, output_layer, img, threshold)

    def get_prediction_images(self,
                              softmax_layer: torch.Tensor,
                              pooled_layer: torch.Tensor,
                              output_layer: torch.Tensor,
                              img: Image.Image,
                              threshold: float) -> HipFractureImage:
        """
        Returns the image with the predicted class and the images of the patches that were used to make the prediction.
        :param softmax_layer:
        :param pooled_layer:
        :param output_layer:
        :param img:
        :param threshold:
        :return:
        """
        patch_size, skip = self.args.get_patch_size()  # Not sure what this does
        prototype_list = []
        sorted_output_layer, sorted_output_layer_indices = torch.sort(output_layer.squeeze(0), descending=True)
        sorted_pooled, sorted_pooled_indices = torch.sort(pooled_layer.squeeze(0), descending=True)

        image = transforms.Resize(size=(self.args.get_arg('image_size'), self.args.get_arg('image_size')))(img)
        img_tensor = transforms.ToTensor()(image).unsqueeze_(0)  # shape (1, 3, h, w)
        # Since the model is trained on 224x224 images, we need to resize the image to that size. Then convert to tensor

        for pred_class_idx in sorted_output_layer_indices:
            # Convention: 0 is fractured, 1 is non-fractured
            if int(pred_class_idx) == 0:
                pred_class = HipFractureEnum.Fractured
            else:
                pred_class = HipFractureEnum.NonFractured

            for prototype_idx in sorted_pooled_indices:
                sim_weight = pooled_layer[0, prototype_idx].item() * self.net.module._classification.weight[
                    pred_class_idx, prototype_idx].item()

                if sim_weight > threshold:
                    # Get the coordinate of the prototype with the highest weight
                    max_h, max_idx_h = torch.max(softmax_layer[0, prototype_idx, :, :], dim=0)
                    max_w, max_idx_w = torch.max(max_h, dim=0)
                    max_idx_h = max_idx_h[max_idx_w].item()
                    max_idx_w = max_idx_w.item()

                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                        self.args.get_arg('image_size'),
                        softmax_layer.shape,
                        patch_size,
                        skip, max_idx_h, max_idx_w)

                    similarity = pooled_layer[0, prototype_idx].item()
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    img_patch = transforms.ToPILImage()(img_tensor_patch)

                    rectangle = ((max_idx_w * skip, max_idx_h * skip), (
                        min(self.args.get_arg('image_size'), max_idx_w * skip + patch_size),
                        min(self.args.get_arg('image_size'), max_idx_h * skip + patch_size)))

                    prototype = HipFracturePrototype(prototype_index=prototype_idx,
                                                     predicted_class=pred_class,
                                                     prototype_image=img_patch,
                                                     coordinates=rectangle,
                                                     similarity_weight=sim_weight,
                                                     similarity=similarity)
                    prototype_list.append(prototype)

        return HipFractureImage(image, prototype_list)
