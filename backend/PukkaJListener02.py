import os
import shutil
from PIL import Image
from mysql.connector import Error
import mysql.connector
from InferenceResult import HipFractureStudy, HipFractureSeries, HipFractureImage, HipFractureEnum
import xml.etree.ElementTree as ElementTree

from PIPNet_inference import PIPNetInference

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(SCRIPT_DIR)
LISTENING_PATH_rela = "pukkaJ-in"
MODEL_DIR_rela = "model/checkpoints/net_trained_last"
LISTENING_PATH = os.path.join(BASE_DIR, LISTENING_PATH_rela)
MODEL_DIR  = os.path.join(BASE_DIR, MODEL_DIR_rela)
STUDIED_PATH=os.path.join(BASE_DIR, 'studied')

ABSTAIN_THRESHOLD = {HipFractureEnum.Fractured: 2.6641653403285863,
                     HipFractureEnum.NonFractured: 3.953101319007754}


class SQLCredentials:
    def __init__(self):
        self.host = "localhost"
        self.user = "root"
        self.password = "0105"


class ImageMetadataXMLParser:
    def __init__(self, file_dir: str):
        tree = ElementTree.parse(file_dir)
        root = tree.getroot()
        for child in root:
            for elem in child:
                match elem.attrib['name']:
                    case "SeriesInstanceUID":
                        self.series_instance_uid = elem.text
                    case "StudyDate":
                        self.study_date = elem.text
                    case "StudyDescription":
                        self.study_description = elem.text
                    case "AccessionNumber":
                        self.accession_number = elem.text
                    case "SeriesDescription":
                        self.series_description = elem.text
                    case "StudyInstanceUID":
                        self.study_instance_uid = elem.text
                    case "SOPInstanceUID":
                        self.image_uid = elem.text


def create_study(image_metadata_list: list, directory: str):
    print("Length of image_metadata_list:", len(image_metadata_list))
    if not image_metadata_list:
        raise ValueError("image_metadata_list is empty")
    return HipFractureStudy(image_metadata_list[0].study_instance_uid,
                            image_metadata_list[0].accession_number,
                            directory,
                            image_metadata_list[0].study_description,
                            image_metadata_list[0].study_date)


def get_image_metadata(directory: str) -> dict:
    image_metadata_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_dir = os.path.join(directory, filename)
            print("Parsing XML file: " + filename)
            image_metadata = ImageMetadataXMLParser(xml_dir)
            image_metadata_dict[filename.strip(".xml")] = image_metadata
    return image_metadata_dict



def create_series_directory(directory: str, image_metadata_list: list):
    for image_metadata in image_metadata_list:
        file_path = os.path.join(directory, image_metadata.series_instance_uid)
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def create_images_directory(directory: str, image_metadata_list: list):
    for image_metadata in image_metadata_list:
        file_path = os.path.join(directory, image_metadata.series_instance_uid, image_metadata.image_uid)
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def create_prototypes_directory(directory: str, image_metadata_list: list):
    for image_metadata in image_metadata_list:
        file_path_non_fractured = os.path.join(directory, image_metadata.series_instance_uid, image_metadata.image_uid,
                                               "non_fractured")
        file_path_fractured = os.path.join(directory, image_metadata.series_instance_uid, image_metadata.image_uid,
                                           "fractured")
        file_path_abstain = os.path.join(directory, image_metadata.series_instance_uid, image_metadata.image_uid,
                                         "abstain")
        if not os.path.exists(file_path_non_fractured):
            os.makedirs(file_path_non_fractured)
        if not os.path.exists(file_path_fractured):
            os.makedirs(file_path_fractured)
        if not os.path.exists(file_path_abstain):
            os.makedirs(file_path_abstain)


def move_images_and_xml_to_image_directory(old_directory: str, new_directory: str, image_metadata_list: list):
    for image_metadata in image_metadata_list:
        old_img_dir = os.path.join(old_directory, image_metadata.image_uid + ".jpg")
        if not os.path.exists(old_img_dir):
            continue
        new_img_dir = os.path.join(new_directory,
                                   image_metadata.series_instance_uid,
                                   image_metadata.image_uid)
        os.makedirs(new_img_dir, exist_ok=True)
        new_img_path = os.path.join(new_img_dir, image_metadata.image_uid + ".jpg")
        shutil.copy(old_img_dir, new_img_path)


def create_series(image_metadata_list: ImageMetadataXMLParser, directory: str):
    return HipFractureSeries(image_metadata_list.series_instance_uid,
                             directory,
                             image_metadata_list.series_description)


def get_study_sql_query(study: HipFractureStudy):
    sql = "INSERT INTO study (study_uid, accession_number, study_directory, " \
          "study_description, predicted_class, RADPEER_score, study_date, study_comment) " \
          "VALUES (%s, %s ,%s, %s, %s, %s, %s, %s)"
    if study.predicted_class == HipFractureEnum.Fractured:
        study_predicted_class = 1
    elif study.predicted_class == HipFractureEnum.NonFractured:
        study_predicted_class = 0
    else:
        study_predicted_class = 2
    val = (study.study_instance_uid, study.accession_number, study.study_dir, study.study_description,
           study_predicted_class, study.RADPEER_score, study.study_date, study.study_comment)
    return sql, val


def get_series_sql_query(series, study_uid):
    sql = "INSERT INTO series (series_uid, series_directory, series_description, study_uid, predicted_class) " \
          "VALUES (%s, %s, %s, %s, %s)"

    if series.predicted_class == HipFractureEnum.Fractured:
        series_predicted_class = 1
    elif series.predicted_class == HipFractureEnum.NonFractured:
        series_predicted_class = 0
    else:
        series_predicted_class = 2

    val = (series.series_instance_uid, series.series_dir, series.series_description, study_uid,
           series_predicted_class)
    return sql, val


def get_images_sql_query(image: HipFractureImage, series_uid):
    if image.predicted_class == HipFractureEnum.Fractured:
        image_predicted_class = 1
    elif image.predicted_class == HipFractureEnum.NonFractured:
        image_predicted_class = 0
    else:
        image_predicted_class = 2

    sql = "INSERT INTO images (image_uid, image_directory, series_uid, predicted_class, " \
          "fractured_annotated_image_directory, non_fractured_annotated_image_directory, " \
          "fractured_sim_weight, non_fractured_sim_weight, manual_annotated_coordinates) " \
          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"

    val = (image.image_uid, image.image_dir, series_uid, image_predicted_class,
           image.fractured_annotated_image_dir, image.non_fractured_annotated_image_dir,
           image.sim_weight_fractured, image.sim_weight_non_fractured, 0)

    return sql, val


def get_prototypes_sql_query(prototype, image_uid):
    if prototype.predicted_class == HipFractureEnum.Fractured:
        prototype_predicted_class = 1
    elif prototype.predicted_class == HipFractureEnum.NonFractured:
        prototype_predicted_class = 0
    else:
        prototype_predicted_class = 2

    sql = "INSERT INTO prototypes (prototype_uid, prototype_image_directory, image_uid, " \
          "sim_weight, similarities, predicted_class, flagged_error, manual," \
          "upper_left_x_coord, upper_left_y_coord, lower_right_x_coord, lower_right_y_coord) " \
          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s ,%s, %s)"

    val = (prototype.prototype_uid, prototype.prototype_image_directory, image_uid,
           prototype.similarity_weight, prototype.similarity,
           prototype_predicted_class, prototype.flagged_error, 0, prototype.coordinates[0][0],
           prototype.coordinates[0][1], prototype.coordinates[1][0], prototype.coordinates[1][1])

    return sql, val


class PukkaJDirectoryObserver:
    def __init__(self, new_directory: str, model_dir=MODEL_DIR, test=False):
        self.new_directory = new_directory
        self.inference_model = PIPNetInference()
        self.inference_model.load_model(model_dir=model_dir)
        self.test = test
        sql_credentials = SQLCredentials()

        self.study_database = mysql.connector.connect(
            host=sql_credentials.host,
            user=sql_credentials.user,
            password=sql_credentials.password
        )

    def image_inference(self, image_dir: str, img_name: str):
        original_image_path = os.path.join(image_dir, img_name)
        image = Image.open(original_image_path)
        hip_fracture_image = self.inference_model.single_image_inference(image)
        hip_fracture_image.save_all_result_images(image_dir, img_name)
        hip_fracture_image.image_dir = os.path.join(image_dir, img_name)
        return hip_fracture_image

    def analyze(self):
        print(f"Observing path: {self.new_directory}")
        for filename in os.listdir(self.new_directory):
            directory = os.path.join(self.new_directory, filename)
            if not os.path.isdir(directory):
                continue
            print(f"Directory detected: {directory}")
            self.process_directory(directory,filename)


    def process_directory(self, directory,filename):
        # Get the metadata for images in the directory
        image_metadata_dict = get_image_metadata(directory)
        study_path = os.path.join(STUDIED_PATH,filename)
        if not os.path.exists(study_path):
            os.makedirs(study_path)
        # Create a study instance using the metadata
        study = create_study(list(image_metadata_dict.values()), directory)

        series_dict = {key: create_series(val, directory) for key, val in image_metadata_dict.items()}
        create_series_directory(study_path, list(image_metadata_dict.values()))
        create_images_directory(study_path, list(image_metadata_dict.values()))
        create_prototypes_directory(study_path, list(image_metadata_dict.values()))
        # Move images and XML files to the image directory
        move_images_and_xml_to_image_directory(directory, study_path, list(image_metadata_dict.values()))

        # Perform inference on each image and add the resulting HipFractureImage to the series
        for key, val in image_metadata_dict.items():
            series_image_dir = os.path.join(study_path, val.series_instance_uid, val.image_uid)
            series_image_name = val.image_uid + ".jpg"
            hip_fracture_image = self.image_inference(series_image_dir, series_image_name)
            series_dict[key].add_image(hip_fracture_image)
        for key, series in series_dict.items():
            study.add_series(series)
        study.get_predicted_class()
        self.write_to_database(study)



    def write_to_database(self, study: HipFractureStudy):
        try:
            cursor = self.study_database.cursor()
            if self.test:
                cursor.execute("USE hip_fracture_study_test")
            else:
                cursor.execute("USE hip_fracture_study")

            sql_study, val_study = get_study_sql_query(study)
            cursor.execute(sql_study, val_study)

            for series in study.series:
                sql_series, val_series = get_series_sql_query(series, study.study_instance_uid)
                cursor.execute(sql_series, val_series)
                for image in series.images:
                    sql_image, val_image = get_images_sql_query(image, series.series_instance_uid)
                    cursor.execute(sql_image, val_image)
                    for prototype in image.prototypes:
                        sql_prototype, val_prototype = get_prototypes_sql_query(prototype, image.image_uid)
                        cursor.execute(sql_prototype, val_prototype)

            self.study_database.commit()

        except mysql.connector.Error as err:
            if err.errno == 1062:  # MySQL error code for duplicate entry
                print(f"Skipping duplicate entry: {err}")
                self.study_database.rollback()  # Rollback the transaction
            else:
                print(f"Error: {err}")
                self.study_database.rollback()  # Rollback the transaction
        finally:
            if 'cursor' in locals():
                cursor.close()


if __name__ == "__main__":
    observer = PukkaJDirectoryObserver(LISTENING_PATH)
    observer.analyze()