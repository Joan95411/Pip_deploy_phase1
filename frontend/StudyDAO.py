import json
from enum import Enum


class HipFractureEnum(Enum):
    """
    An inference can be classified as fractured, non-fractured or abstain.
    From the model: Fractured = 0, NonFractured = 1
    """
    Fractured = "Fractured"
    NonFractured = "Not Fractured"
    Abstain = "Abstain"


class StudyDAO:
    def __init__(self, study_instance_uid: str,
                 accession_number: str,
                 study_dir: str,
                 study_description: str,
                 predicted_class: int,
                 study_date: str,
                 RADPEER_score: int = None,
                 study_comment: str = None):
        self.accession_number = accession_number
        self.study_instance_uid = study_instance_uid
        self.study_dir = study_dir
        self.study_description = study_description

        if predicted_class == 0:
            self.predicted_class = HipFractureEnum.NonFractured
        elif predicted_class == 1:
            self.predicted_class = HipFractureEnum.Fractured
        else:
            self.predicted_class = HipFractureEnum.Abstain

        self.RADPEER_score = RADPEER_score
        self.study_comment = study_comment
        self.study_date = study_date
        self.series = []
        self.weight_fractured = None
        self.weight_non_fractured = None

    def add_series(self, series):
        self.series.append(series)

    def get_series_by_uid(self, series_uid: str):
        for series in self.series:
            if series.series_uid == series_uid:
                return series


class SeriesDAO:
    def __init__(self, series_uid: str,
                 study_instance_uid: str,
                 series_dir: str,
                 series_description: str,
                 predicted_class: int,
                 ):
        self.series_uid = series_uid
        self.study_instance_uid = study_instance_uid
        self.series_dir = series_dir
        self.series_description = series_description
        if predicted_class == 0:
            self.predicted_class = HipFractureEnum.NonFractured
        elif predicted_class == 1:
            self.predicted_class = HipFractureEnum.Fractured
        else:
            self.predicted_class = HipFractureEnum.Abstain
        self.images = []
        self.weight_fractured = None
        self.weight_non_fractured = None

    def add_image(self, image):
        self.images.append(image)

    def get_series_by_uid(self, image_uid: str):
        for image in self.images:
            if image.image_uid == image_uid:
                return image


class ImageDAO:
    def __init__(self, image_uid: str,
                 series_uid: str,
                 image_dir: str,
                 fractured_annotated_image_dir: str,
                 non_fractured_annotated_image_dir: str,
                 fractured_sim_weight: float,
                 non_fractured_sim_weight: float,
                 manually_annotated: bool,
                 predicted_class: int,
                 ):
        self.image_uid = image_uid
        self.series_uid = series_uid
        self.image_dir = image_dir
        if predicted_class == 0:
            self.predicted_class = HipFractureEnum.NonFractured
        elif predicted_class == 1:
            self.predicted_class = HipFractureEnum.Fractured
        else:
            self.predicted_class = HipFractureEnum.Abstain
        self.fractured_sim_weight = fractured_sim_weight
        self.non_fractured_sim_weight = non_fractured_sim_weight

        self.fractured_annotated_image_dir = fractured_annotated_image_dir
        self.non_fractured_annotated_image_dir = non_fractured_annotated_image_dir
        self.manually_annotated = manually_annotated

        self.prototypes = []

    def add_prototype(self, prototype):
        self.prototypes.append(prototype)

    def prototypes_to_json(self):
        prototypes_json = []
        for prototype in self.prototypes:
            if prototype.predicted_class == self.predicted_class:
                prototypes_json.append(prototype.to_dict())
        return json.dumps(prototypes_json)


class PrototypeDAO:
    def __init__(self, prototype_uid: str,
                 image_uid: str,
                 prototype_dir: str,
                 predicted_class: int,
                 prototype_index,
                 coordinates: tuple[tuple[int, int], tuple[int, int]],
                 similarity_weight,
                 similarity,
                 flagged_error: bool,
                 manually_annotated: bool):
        self.prototype_uid = prototype_uid
        self.image_uid = image_uid
        self.prototype_dir = prototype_dir
        self.prototype_index = prototype_index
        self.coordinates = coordinates
        self.similarity_weight = similarity_weight
        self.similarity = similarity
        self.flagged_error = flagged_error
        self.manually_annotated = manually_annotated
        if predicted_class == 0:
            self.predicted_class = HipFractureEnum.NonFractured
        elif predicted_class == 1:
            self.predicted_class = HipFractureEnum.Fractured
        else:
            self.predicted_class = HipFractureEnum.Abstain

    def to_dict(self):
        return {
            "prototype_index": self.prototype_index,
            "prototype_uid": self.prototype_uid,
            "coordinates": self.coordinates,
            "similarity_weight": self.similarity_weight,
            "similarity": self.similarity,
            "predicted_class": self.predicted_class.value,
            "flagged_error": self.flagged_error,
            "manually_annotated": self.manually_annotated,
        }
