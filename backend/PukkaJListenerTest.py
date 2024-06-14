import os
import shutil
import unittest
from watchdog.observers import Observer
from PukkaJListener import PukkaJDirectoryObserver, SQLCredentials
import mysql.connector
from mysql.connector import OperationalError, ProgrammingError


def executeScriptsFromFile(filename, study_db):
    cursor = study_db.cursor()
    # Open and read the file as a single buffer

    fd = open(filename, 'r')
    sqlFile = fd.read()
    fd.close()

    # all SQL commands (split on ';')
    sqlCommands = sqlFile.split(';')

    # Execute every command from the input file
    for command in sqlCommands:
        # This will skip and report errors
        # For example, if the tables do not yet exist, this will skip over
        # the DROP TABLE commands
        try:
            cursor.execute(command)
        except OperationalError as msg:
            print("Command skipped: ", msg)
        except ProgrammingError as msg:
            print("Command skipped: ", msg)
    cursor.execute("USE hip_fracture_study_test")
    study_db.commit()


class TestPukkaJListener(unittest.TestCase):

    def setUp(self):
        sql_credentials = SQLCredentials()

        self.study_database = mysql.connector.connect(
            host=sql_credentials.host,
            user=sql_credentials.user,
            password=sql_credentials.password
        )

        executeScriptsFromFile("database/db-test-schema.sql", self.study_database)
        self.test_dir = "TestFolder/"
        shutil.rmtree(self.test_dir, ignore_errors=True)

        self.pukkaJ_test_dir = os.path.join(self.test_dir, "pukkaJ-in/")
        self.studied_dir = os.path.join(self.test_dir, "studied/")

        os.makedirs(self.pukkaJ_test_dir, exist_ok=True)
        os.makedirs(self.studied_dir, exist_ok=True)

        self.event_handler = PukkaJDirectoryObserver(self.studied_dir, test=True)
        self.observer = Observer()

        self.observer.schedule(self.event_handler, self.pukkaJ_test_dir, recursive=True)
        self.observer.start()

    def tearDown(self):
        self.observer.stop()
        self.observer.join()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        cursor = self.study_database.cursor()
        cursor.execute("DROP DATABASE IF EXISTS hip_fracture_study_test")
        self.study_database.commit()

    def test_listening_new_file(self):
        self.study_to_analyse = "1.2.276.0.50.10030050031.14138620.14313864.148504"
        pukkaJ_study_to_analyse = os.path.join(self.pukkaJ_test_dir, self.study_to_analyse)
        study_for_test = os.path.join("inference_in/" + self.study_to_analyse)
        shutil.copytree(study_for_test, pukkaJ_study_to_analyse, dirs_exist_ok=True)

        for study in os.listdir(self.studied_dir):
            assert study == self.study_to_analyse

            series_dir = os.listdir(os.path.join(self.studied_dir, study))
            assert len(series_dir) == 2

            for series in series_dir:
                images_dir = os.listdir(os.path.join(self.studied_dir, study, series))
                assert len(images_dir) == 1


if __name__ == "__main__":
    os.chdir("../")
    unittest.main()
