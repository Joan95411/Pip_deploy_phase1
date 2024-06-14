import os
import shutil

from PIL import Image
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import time

from InferenceResult import HipFractureStudy, HipFractureSeries, HipFractureImage, HipFractureEnum
import xml.etree.ElementTree as ElementTree

from PIPNet_inference import PIPNetInference
import mysql.connector

LISTENING_PATH = "pukkaJ-in"
MODEL_DIR = "model/checkpoints/net_trained_last"
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
    return HipFractureStudy(image_metadata_list[0].study_instance_uid,
                            image_metadata_list[0].accession_number,
                            directory,
                            image_metadata_list[0].study_description,
                            image_metadata_list[0].study_date)


def get_image_metadata(directory: str) -> dict:
    """
    :param directory: The original directory of the study
    """
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
                                   image_metadata.image_uid,
                                   image_metadata.image_uid + ".jpg")
        shutil.move(old_img_dir, new_img_dir)


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


class PukkaJDirectoryObserver(FileSystemEventHandler):
    def __init__(self, new_directory: str, model_dir=MODEL_DIR, test=False):
        """
        :param new_directory: The directory where the new study folders will be created
        :param model_dir: The directory where the model is stored
        :param test: If true, the test database will be used
        """
        super().__init__()
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
        return hip_fracture_image

    def add_study_to_database(self, study: HipFractureStudy):
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

    def on_created(self, event):
        if event.is_directory:
            print(f"Directory modified: {event.src_path}")
            study_folder = event.src_path
            time.sleep(2)
            new_study_folder = os.path.join(self.new_directory, study_folder.split('/')[-1])
            image_metadata_dict = get_image_metadata(study_folder)
            image_metadata_list = list(image_metadata_dict.values())
            img_fracture_study = create_study(image_metadata_list, new_study_folder)
            create_series_directory(new_study_folder, image_metadata_list)
            create_images_directory(new_study_folder, image_metadata_list)
            create_prototypes_directory(new_study_folder, image_metadata_list)
            move_images_and_xml_to_image_directory(study_folder, new_study_folder, image_metadata_list)
            series_list = os.listdir(new_study_folder)
            print("Created and moved images to new study folder", new_study_folder)
            for i, series_dir in enumerate(series_list):
                if series_dir.startswith('.'):
                    continue
                full_series_dir = os.path.join(new_study_folder, series_dir)
                img_fracture_series = HipFractureSeries(image_metadata_list[i].series_instance_uid,
                                                        series_dir,
                                                        image_metadata_list[i].series_description)
                image_list = os.listdir(full_series_dir)
                for image_dir in image_list:
                    print("Inference image: " + image_dir)
                    img_folder = os.path.join(full_series_dir, image_dir)
                    hip_fracture_img = self.image_inference(image_dir=img_folder, img_name=image_dir + ".jpg")
                    img_fracture_series.add_image(hip_fracture_img)
                img_fracture_study.add_series(img_fracture_series)

            shutil.rmtree(study_folder)
            print("Removed original study folder")
            img_fracture_study.get_predicted_class()
            self.add_study_to_database(img_fracture_study)
            print("Added study to database. Finishing up...")

    def stop_mysql(self):
        self.study_database.close()


if __name__ == "__main__":
    os.chdir("../")
    os.makedirs("studied/", exist_ok=True)
    if not os.path.exists(LISTENING_PATH):
        print("Listening directory not found.")
        exit(1)
    if not os.path.exists(MODEL_DIR):
        print("Model directory not found. Please follow instructions in README.md to download the model.")
        exit(1)

    event_handler = PukkaJDirectoryObserver("studied")
    observer = Observer()
    observer.schedule(event_handler, LISTENING_PATH, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        event_handler.stop_mysql()
    observer.join()
