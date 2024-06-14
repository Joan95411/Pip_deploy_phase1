DROP DATABASE IF EXISTS hip_fracture_study_test;
CREATE DATABASE IF NOT EXISTS hip_fracture_study_test;

USE hip_fracture_study_test;
DROP TABLE IF EXISTS prototypes;
DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS series;
DROP TABLE IF EXISTS study;

CREATE TABLE IF NOT EXISTS study (
                     study_uid VARCHAR(300) PRIMARY KEY,
                     accession_number VARCHAR(300) NOT NULL,
                     study_directory VARCHAR(1000) NOT NULL,
                     study_description VARCHAR(300) NOT NULL,
                     predicted_class BIT(2) NOT NULL,
                     RADPEER_score INT CHECK (RADPEER_score >= 1 AND RADPEER_score <= 5),
                     study_date DATE NOT NULL,
                     study_comment VARCHAR(1000)
);

CREATE TABLE IF NOT EXISTS series (
                     series_uid VARCHAR(300) PRIMARY KEY,
                     study_uid VARCHAR(300) NOT NULL,
                     series_directory VARCHAR(1000) NOT NULL,
                     series_description VARCHAR(300) NOT NULL,
                     predicted_class BIT(2) NOT NULL,
                     FOREIGN KEY (study_uid) REFERENCES study(study_uid)
);

CREATE TABLE IF NOT EXISTS images (
                     image_uid VARCHAR(300) PRIMARY KEY,
                     series_uid VARCHAR(300) NOT NULL,
                     image_directory VARCHAR(1000) NOT NULL,
                     fractured_annotated_image_directory VARCHAR(1000) NOT NULL,
                     non_fractured_annotated_image_directory VARCHAR(1000) NOT NULL,
                     fractured_sim_weight FLOAT NOT NULL,
                     non_fractured_sim_weight FLOAT NOT NULL,
                     predicted_class BIT(2) NOT NULL,
                     manual_annotated_coordinates VARCHAR(1),
                     FOREIGN KEY (series_uid) REFERENCES series(series_uid)
);

CREATE TABLE IF NOT EXISTS prototypes (
                     prototype_uid VARCHAR(500) PRIMARY KEY,
                     image_uid VARCHAR(300) NOT NULL,
                     prototype_image_directory VARCHAR(1000) NOT NULL,
                     upper_left_x_coord INT NOT NULL,
                     upper_left_y_coord INT NOT NULL,
                     lower_right_x_coord INT NOT NULL,
                     lower_right_y_coord INT NOT NULL,
                     sim_weight FLOAT NOT NULL,
                     similarities FLOAT NOT NULL,
                     predicted_class BIT(2) NOT NULL,
                     flagged_error BIT(1) NOT NULL,
                     manual BIT(1) NOT NULL,
                     FOREIGN KEY (image_uid) REFERENCES images(image_uid)
);



