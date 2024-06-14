import os
from abc import ABC
from enum import Enum
import random
from json import JSONEncoder
from typing import List
from PIL import Image, ImageDraw
from itertools import cycle
import json

MANUAL_PROTOTYPE_COUNTER = 1000


def get_manual_prototype_counter():
    """
    Automatic prototype have id 0-999, manual prototype have id 1000+
    :return: id for manual prototype
    """
    global MANUAL_PROTOTYPE_COUNTER
    MANUAL_PROTOTYPE_COUNTER += 1
    return MANUAL_PROTOTYPE_COUNTER


class HipFractureEnum(Enum):
    """
    An inference can be classified as fractured, non-fractured or abstain.
    From the model: Fractured = 0, NonFractured = 1
    """
    Fractured = "Fractured"
    NonFractured = "Not Fractured"
    Abstain = "Abstain"


ABSTAIN_THRESHOLD = {HipFractureEnum.Fractured: 2.6641653403285863,
                     HipFractureEnum.NonFractured: 3.953101319007754}


class AbstractHipFracturePrototype(ABC):
    """
    Abstract class for a prototype
    """

    def __init__(self, predicted_class: HipFractureEnum,
                 prototype_image: Image,
                 coordinates: tuple[tuple[int, int], tuple[int, int]]):
        self.predicted_class = predicted_class
        self.prototype_image = prototype_image
        self.coordinates = coordinates
        self.prototype_image_directory = None
        self.prototype_uid = None
        self.prototype_index = None

    def save_prototype_image(self, parent_image_dir: str, image_name: str):
        """
        Save the prototype image to the parent image directory
        :param parent_image_dir: Parent image directory
        :param image_name: Name of the image
        """
        if self.predicted_class == HipFractureEnum.Fractured:
            self.prototype_image_directory = os.path.join(parent_image_dir,
                                                          "fractured",
                                                          # image_name.strip(".jpg") +
                                                          "prototype_" + str(self.prototype_index) + ".jpg")
            self.prototype_image.save(self.prototype_image_directory)
        elif self.predicted_class == HipFractureEnum.NonFractured:
            self.prototype_image_directory = os.path.join(parent_image_dir,
                                                          "non_fractured",
                                                          # image_name.strip(".jpg") +
                                                          "prototype_" + str(self.prototype_index) + ".jpg")
            self.prototype_image.save(self.prototype_image_directory)
        else:  # Abstain case
            self.prototype_image_directory = os.path.join(parent_image_dir,
                                                          "abstain",
                                                          # image_name.strip(".jpg") +
                                                          "prototype_" + str(self.prototype_index) + ".jpg")
            self.prototype_image.save(self.prototype_image_directory)

        self.prototype_uid = image_name + "_" + str(self.prototype_index)


class ManualHipFracturePrototype(AbstractHipFracturePrototype):
    """
    Class for a manual prototype
    """

    def __init__(self, predicted_class: HipFractureEnum,
                 prototype_image: Image,
                 coordinates: tuple[tuple[int, int], tuple[int, int]],
                 ):
        """
        :param predicted_class: Predicted class of the prototype
        :param prototype_image: Image of the prototype
        :param coordinates: Coordinates of the prototype in the original image
        """
        super().__init__(predicted_class, prototype_image, coordinates)
        self.prototype_index = get_manual_prototype_counter()
        self.prototype_index = None

    def save_prototype_image(self, parent_image_dir: str, image_name: str):
        """
        Save the prototype image to the parent image directory
        :param parent_image_dir: Parent image directory
        :param image_name: Name of the image
        """
        if self.predicted_class == HipFractureEnum.Fractured:
            self.prototype_image_directory = os.path.join(parent_image_dir,
                                                          "fractured", image_name.strip(".jpg") +
                                                          "-prototype_manual" + ".jpg")
            self.prototype_image.save(self.prototype_image_directory)
        elif self.predicted_class == HipFractureEnum.NonFractured:
            self.prototype_image_directory = os.path.join(parent_image_dir,
                                                          "non_fractured", image_name.strip(".jpg") +

                                                          "-prototype_manual" + ".jpg")
            self.prototype_image.save(self.prototype_image_directory)


class HipFracturePrototype(AbstractHipFracturePrototype):
    def __init__(self,
                 prototype_index,
                 predicted_class: HipFractureEnum,
                 prototype_image: Image,
                 coordinates: tuple[tuple[int, int], tuple[int, int]],
                 similarity_weight,
                 similarity):
        """
        :param prototype_index: Index of the prototype in the prototype set
        :param predicted_class: Predicted class of the prototype
        :param prototype_image: Image of the prototype
        :param coordinates: Coordinates of the prototype in the original image
        :param similarity_weight: Weight of the prototype
        :param similarity: Similarity of the prototype to the original image
        """
        super().__init__(predicted_class, prototype_image, coordinates)
        self.prototype_index = int(prototype_index)
        self.similarity_weight = similarity_weight
        self.similarity = similarity
        self.prototype_image_directory = None
        self.flagged_error = False
        if self.similarity_weight < ABSTAIN_THRESHOLD[predicted_class]:
            self.predicted_class = HipFractureEnum.Abstain


class HipFractureImage:
    def __init__(self,
                 original_image: Image.Image,
                 prototypes: List[HipFracturePrototype]):
        """
        :param original_image: Original image
        :param prototypes: List of prototypes belonging to the image
        """
        self.prototypes = prototypes
        self.original_image = original_image
        self.fractured_image = self.__draw_prototypes__(HipFractureEnum.Fractured)
        self.non_fractured_image = self.__draw_prototypes__(HipFractureEnum.NonFractured)
        self.sim_weight_fractured, self.sim_weight_non_fractured = self.__get_sim_weights__()
        self.predicted_class = self.__get_predicted_class__()
        self.image_dir = None
        self.fractured_annotated_image_dir = None
        self.non_fractured_annotated_image_dir = None
        self.manually_annotated_prototype = None
        self.image_uid = None

    def manually_annotate_prototype(self, coordinates, prototype_class: HipFractureEnum):
        """
        manually annotate a prototype
        :param coordinates: coordinates of the prototype
        :param prototype_class: class of the prototype
        """
        self.manually_annotated_prototype = ManualHipFracturePrototype(
            predicted_class=prototype_class,
            prototype_image=None,
            coordinates=coordinates,
        )

    def __get_predicted_class__(self):
        """
        :return: predicted class of the image
        """
        if self.__check_abstain__():
            return HipFractureEnum.Abstain
        else:
            return HipFractureEnum.Fractured \
                if self.sim_weight_fractured > self.sim_weight_non_fractured \
                else HipFractureEnum.NonFractured

    def __check_abstain__(self):
        """
        :return: True if all prototypes are abstain or there is no prototype, False otherwise
        """
        for prototype in self.prototypes:
            if prototype.predicted_class != HipFractureEnum.Abstain:
                return False

        return True

    def __get_sim_weights__(self):
        sim_weight_fractured = 0
        sim_weight_non_fractured = 0

        for prototype in self.prototypes:
            if prototype.predicted_class == HipFractureEnum.Fractured:
                sim_weight_fractured += prototype.similarity_weight
            else:
                sim_weight_non_fractured += prototype.similarity_weight
        return sim_weight_fractured, sim_weight_non_fractured

    def get_prototypes(self, fracture_class: HipFractureEnum):
        prototypes = []
        for prototype in self.prototypes:
            if prototype.predicted_class == fracture_class:
                prototypes.append(prototype)
        return prototypes

    def __draw_prototypes__(self, fracture_class: HipFractureEnum):
        image = self.original_image.copy()
        draw = ImageDraw.Draw(image)
        for prototype in self.get_prototypes(fracture_class):
            color = 'white'
            draw.point(prototype.coordinates[0], fill=color)
            draw.point(prototype.coordinates[1], fill=color)
            draw.rectangle(prototype.coordinates, outline=color, width=2)
        return image

    def get_prototype_images(self, fracture_class: HipFractureEnum):
        prototype_images = []
        for prototype in self.get_prototypes(fracture_class):
            prototype_images.append(prototype.prototype_image)
        return prototype_images

    def save_all_result_images(self, image_dir: str, img_name: str):
        img_name = img_name.strip(".jpg")
        self.image_uid = img_name
        self.image_dir = image_dir + img_name + ".jpg"

        self.fractured_annotated_image_dir = os.path.join(image_dir, "fractured",
                                                          img_name + "-fractured.jpg")
        self.non_fractured_annotated_image_dir = os.path.join(image_dir, "non_fractured",
                                                              img_name + "-non_fractured.jpg")
        self.fractured_image.save(self.fractured_annotated_image_dir)
        self.non_fractured_image.save(self.non_fractured_annotated_image_dir)
        for prototype in self.prototypes:
            prototype.save_prototype_image(image_dir, img_name)


class HipFractureSeries:
    def __init__(self, series_instance_uid: str, series_dir: str, series_description: str):
        self.series_instance_uid = series_instance_uid
        self.series_dir = series_dir
        self.series_description = series_description
        self.predicted_class = HipFractureEnum.NonFractured
        self.images = []
        self.weight_fractured = None
        self.weight_non_fractured = None

    def add_image(self, image: HipFractureImage):
        self.images.append(image)

    def get_predicted_class(self):
        """
        Get the predicted class of the series.
        Predicted fracture when at least one image is predicted as fractured.
        Predicted non-fracture/abstain when all images are predicted as non-fractured/abstain
        """
        abstain = True
        for image in self.images:
            if image.predicted_class != HipFractureEnum.Abstain:
                abstain = False
                if image.predicted_class == HipFractureEnum.Fractured:
                    self.predicted_class = HipFractureEnum.Fractured
                    return
        if abstain:
            self.predicted_class = HipFractureEnum.Abstain


class HipFractureStudy:
    def __init__(self, study_instance_uid: str,
                 accession_number: str,
                 study_dir: str,
                 study_description: str,
                 study_date: str):
        self.accession_number = accession_number
        self.study_instance_uid = study_instance_uid
        self.study_dir = study_dir
        self.study_description = study_description
        self.predicted_class = HipFractureEnum.NonFractured
        self.RADPEER_score = None
        self.study_comment = None
        self.study_date = study_date
        self.series = []

    def add_series(self, series: HipFractureSeries):
        self.series.append(series)

    def get_predicted_class(self):
        abstain = True
        for series in self.series:
            series.get_predicted_class()
            if series.predicted_class != HipFractureEnum.Abstain:
                abstain = False
                if series.predicted_class == HipFractureEnum.Fractured:
                    self.predicted_class = HipFractureEnum.Fractured
                    return
        if abstain:
            self.predicted_class = HipFractureEnum.Abstain
