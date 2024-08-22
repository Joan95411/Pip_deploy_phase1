import os

from flask import Flask, request, render_template, send_from_directory, Response, redirect, url_for
import mysql.connector
from SQLCredentials import SQLCredentials
from StudyDAO import StudyDAO, SeriesDAO, ImageDAO, PrototypeDAO
import logging
import traceback


logger = logging.getLogger("webapp")
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(SCRIPT_DIR)



app = Flask(__name__)
app.secret_key = "superSecretKey"

sql_credentials = SQLCredentials()

study_database = mysql.connector.connect(
    host=sql_credentials.host,
    user=sql_credentials.user,
    password=sql_credentials.password
)


class NoStudyException(Exception):
    pass


class NoSeriesException(Exception):
    pass


class NoImageException(Exception):
    pass

@app.route('/')
def homepage():
    page = request.args.get('page', 1, type=int)
    studyDAOs = fetch_page_studies(page)
    return render_template('all-study-view.html', studies=studyDAOs,page=page)

def get_study_dao(study):
    studyDAO = StudyDAO(
        study_instance_uid=study[0],
        accession_number=study[1],
        study_dir=study[2],
        study_description=study[3],
        predicted_class=study[4],
        study_date=study[6],
    )

    if study[7]:
        studyDAO.study_comment = study[7]
    if study[5]:
        studyDAO.RADPEER_score = study[5]

    return studyDAO


def fetch_study(cursor, accessionNumber) -> StudyDAO:
    sql = "SELECT * FROM study WHERE accession_number = %s"
    val = (accessionNumber,)
    cursor.execute(sql, val)
    study = cursor.fetchall()
    if not study:
        raise NoStudyException
    studyDAO = get_study_dao(study[0])
    return studyDAO


def fetch_series(cursor, studyDAO):
    sql = "SELECT * FROM series WHERE study_uid = %s"
    val = (studyDAO.study_instance_uid,)
    cursor.execute(sql, val)
    series = cursor.fetchall()
    if not series:
        raise NoSeriesException
    return series


def get_series_DAO(s) -> SeriesDAO:
    return SeriesDAO(
        series_uid=s[0],
        study_instance_uid=s[1],
        series_dir=s[2],
        series_description=s[3],
        predicted_class=s[4],
    )


def fetch_images(cursor, seriesDAO):
    sql = "SELECT * FROM images WHERE series_uid = %s"
    val = (seriesDAO.series_uid,)
    cursor.execute(sql, val)
    images = cursor.fetchall()
    if not images:
        raise NoImageException
    return images


def get_image_DAO(i):
    return ImageDAO(
        image_uid=i[0],
        series_uid=i[1],
        image_dir=i[2],
        fractured_annotated_image_dir=i[3],
        non_fractured_annotated_image_dir=i[4],
        fractured_sim_weight=i[5],
        non_fractured_sim_weight=i[6],
        predicted_class=i[7],
        manually_annotated=False if i[8] == 0 else True,
    )


def fetch_prototypes(cursor, imageDAO):
    sql = "SELECT * FROM prototypes WHERE image_uid = %s"
    val = (imageDAO.image_uid,)
    cursor.execute(sql, val)
    return cursor.fetchall()


def get_prototype_DAO(p):
    coordinates = ((p[3], p[4]), (p[5], p[6]))

    return PrototypeDAO(
        prototype_uid=p[0],
        prototype_index=p[0].split("_")[-1],
        image_uid=p[1],
        prototype_dir=p[2],
        coordinates=coordinates,
        similarity_weight=p[7],
        similarity=p[8],
        predicted_class=p[9],
        flagged_error=p[10],
        manually_annotated=p[11]
    )


def fetch_all_studies():
    cursor = study_database.cursor(buffered=True)
    cursor.execute("USE hip_fracture_study")
    sql = "SELECT * FROM study"
    cursor.execute(sql)
    studies = cursor.fetchall()
    studyDAOs = []
    for study in studies:
        studyDAOs.append(get_study_dao(study))
    study_database.commit()
    return studyDAOs

def fetch_page_studies(page):

    per_page = 4  # Number of items per page
    offset = (page-1) * per_page  # Calculate the offset
    cursor = study_database.cursor(buffered=True)
    cursor.execute("USE hip_fracture_study")

    cursor.execute("SELECT * FROM study LIMIT %s OFFSET %s", (per_page, offset))
    studies = cursor.fetchall()

    studyDAOs = []
    for study in studies:
        studyDAOs.append(get_study_dao(study))

    study_database.commit()
    return studyDAOs
    
@app.route('/study/')
def redirect_to_study():
    accessionNumber = request.args.get('accessionNumber')

    if not accessionNumber:
        return "Accession number is required", 400

    # Redirect to the original route
    return redirect(url_for('view_single_study', accessionNumber=accessionNumber))
    
@app.route('/study/<accessionNumber>')
def view_single_study(accessionNumber):
    cursor = study_database.cursor(buffered=True)
    cursor.execute("USE hip_fracture_study")
    try:
        studyDAO = fetch_study(cursor, accessionNumber)
    except NoStudyException:
        return "No study found with accession number " + accessionNumber

    try:
        series = fetch_series(cursor, studyDAO)
    except NoSeriesException:
        return "No series found for study with accession number " + accessionNumber

    for s in series:
        seriesDAO = get_series_DAO(s)
        try:
            images = fetch_images(cursor, seriesDAO)
        except NoImageException:
            return "No image found for series with uid " + seriesDAO.series_uid
        for i in images:
            imageDAO = get_image_DAO(i)
            prototypes = fetch_prototypes(cursor, imageDAO)
            for p in prototypes:
                prototypeDAO = get_prototype_DAO(p)
                imageDAO.add_prototype(prototypeDAO)
            seriesDAO.add_image(imageDAO)
        studyDAO.add_series(seriesDAO)

    return render_template('single-study-view.html', study=studyDAO)


@app.route("/study/all")
def browse_all_study():
    page = request.args.get('page', 1, type=int)
    studyDAOs = fetch_page_studies(page)
    return render_template('all-study-view.html', studies=studyDAOs,page=page)


@app.route('/image/<path:path>')
def serve_image(path):
    if not path.startswith('//'):
        path = '/' + path
    #6003156567
    # Print the final path for debugging
    print("Final image path:", path)

    # Serve the file using the correct directory and file name
    return send_from_directory(os.path.dirname(path), os.path.basename(path))
    # image_path = path

    # print(image_path)
    # return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))


@app.route("/study/flag_error/<prototype_uid>", methods=['PATCH'])
def patch(prototype_uid):
    cursor = study_database.cursor(buffered=True)
    cursor.execute("USE hip_fracture_study")
    sql = "UPDATE prototypes SET flagged_error = 1 WHERE prototype_uid = %s"
    val = (prototype_uid,)
    try:
        cursor.execute(sql, val)
        study_database.commit()
    except Exception as e:
        print("An error occurred:", e)
        study_database.rollback()
        return Response(status=500)
    return Response(status=200)



@app.route('/study/<accessionNumber>/save', methods=["POST"])
def save_comment(accessionNumber):
    cursor = study_database.cursor(buffered=True)
    cursor.execute("USE hip_fracture_study")
    if 'comment' in request.form:
        sql = "UPDATE study SET study_comment = %s WHERE accession_number = %s"
        val = (request.form['comment'], accessionNumber,)
        cursor.execute(sql, val)

    sql = "UPDATE study SET RADPEER_score = %s WHERE accession_number = %s"
    val = (request.form['RADPEER_score'], accessionNumber,)
    try:
        cursor.execute(sql, val)
        study_database.commit()
    except Exception as e:
        print("An error occurred in the process of ", e)
        study_database.rollback()
        return Response(status=500)
    return Response(status=200)


if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0',port=5000,debug=True)
    except KeyboardInterrupt:
        study_database.close()
