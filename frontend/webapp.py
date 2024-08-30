import os
from flask import Flask, request, render_template, send_from_directory, Response, redirect, url_for, jsonify
import mysql.connector
from SQLCredentials import SQLCredentials
from StudyDAO import StudyDAO, SeriesDAO, ImageDAO, PrototypeDAO
import logging
import base64
from io import BytesIO
from PIL import Image
import json
import cv2
import numpy as np
import threading


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

@app.route('/study/<accessionNumber>/previous')
def previous_study(accessionNumber):
    cursor = study_database.cursor(buffered=True)
    sql = """
        SELECT accession_number 
        FROM study 
        WHERE accession_number < %s
        ORDER BY accession_number DESC
        LIMIT 1;"""
    val = (accessionNumber,)
    cursor.execute(sql, val)
    previous_study=cursor.fetchone()
    if previous_study:
        return redirect(url_for('view_single_study', accessionNumber=previous_study[0]))
    else:
        # Handle case where there is no next study (perhaps redirect to the first study or show a message)
        return redirect(url_for('view_single_study', accessionNumber=accessionNumber))

@app.route('/study/<accessionNumber>/next')
def next_study(accessionNumber):
    cursor = study_database.cursor(buffered=True)
    sql = """
        SELECT accession_number 
        FROM study 
        WHERE accession_number > %s
        ORDER BY accession_number ASC
        LIMIT 1;"""
    val = (accessionNumber,)
    cursor.execute(sql, val)
    next_study=cursor.fetchone()
    if next_study:
        return redirect(url_for('view_single_study', accessionNumber=next_study[0]))
    else:
        # Handle case where there is no next study (perhaps redirect to the first study or show a message)
        return redirect(url_for('view_single_study', accessionNumber=accessionNumber))
        
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

def get_annotation_DAO(s) -> AnnotationDAO:
    return AnnotationDAO(
        annotation_id=s[0],
        annotation_dir=s[1],
        source_dir=s[2],
        author=s[3],
        image_uid=s[4],
        points=s[5],
        annotation_comment=s[6],
        annotation_status=s[7],
        created_at=s[8]
    )
    
def fetch_all_studies():
    cursor = study_database.cursor(buffered=True)
    cursor.execute("USE hip_fracture_study")
    sql = "SELECT * FROM study ORDER BY accession_number ASC"
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

    cursor.execute("SELECT * FROM study ORDER BY accession_number ASC LIMIT %s OFFSET %s", (per_page, offset))
    studies = cursor.fetchall()

    studyDAOs = []
    for study in studies:
        studyDAOs.append(get_study_dao(study))

    study_database.commit()
    return studyDAOs

@app.route('/fetch_annotations', methods=['GET'])
def fetch_annotations():
    cursor = study_database.cursor(buffered=True)
    image_uid = request.args.get('image_uid')

    if not image_uid:
        return jsonify({'error': 'image_uid is required'}), 400

    sql = "SELECT * FROM annotations WHERE image_uid = %s"
    val = (image_uid,)
    cursor.execute(sql, val)
    annotations = cursor.fetchall()

    # Convert annotations to a list of dictionaries
    annotations_list = [
        {'id': get_annotation_DAO(a).annotation_id,
         'dir': get_annotation_DAO(a).annotation_dir,
         'author': get_annotation_DAO(a).author,
         'comment': get_annotation_DAO(a).annotation_comment}
        for a in annotations
    ]

    return jsonify(annotations_list), 200

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

@app.route('/save-drawing2', methods=['POST'])
def save_drawing2():
    data = request.json
    image_data = data['image']
    file_name = data['name']
    save_directory = data['directory']
    clicks = data['points']
    image_uid=data['image_uid']
    image_data = image_data.split(',')[1]  # Remove the data:image/png;base64, part
    image_bytes = base64.b64decode(image_data)

    # Save the image
    try:
        os.makedirs(save_directory, exist_ok=True)  # Create the directory if it does not exist
        image = Image.open(BytesIO(image_bytes))
        sdir=os.path.join(save_directory, file_name )
        image.save(sdir)
        # Save the points (clicks) to the database
        cursor = study_database.cursor()
        cursor.execute("USE hip_fracture_study")
        author = data.get('author', 'unknown')  # Replace with actual author data as needed
        annotation_status = 'annotated'  # Default status

        cursor.execute("""
            INSERT INTO annotations (dir, author, image_uid, points, annotation_status) 
            VALUES ( %s, %s, %s, %s, %s)
        """, (sdir, author, image_uid, json.dumps(clicks), annotation_status))

        study_database.commit()
        cursor.close()

        return jsonify({"success": True})
    except Exception as e:
        print(f"Error saving to database: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/save-drawing3', methods=['POST'])
def save_drawing3():
    data = request.json
    file_name = data['name']
    save_directory = data['directory']
    clicks = data['points']
    image_uid=data['image_uid']
    source_dir=data['source_dir']
    try:

        sdir=os.path.join(save_directory, file_name )
        cursor = study_database.cursor()
        author = data.get('author', 'unknown')  # Replace with actual author data as needed
        annotation_status = 'annotated'  # Default status
        cursor.execute("USE hip_fracture_study")
        cursor.execute("""
            INSERT INTO annotations (dir,source_dir, author, image_uid, points, annotation_status) 
            VALUES ( %s, %s, %s, %s, %s, %s)
        """, (sdir, source_dir,author, image_uid, json.dumps(clicks), annotation_status))
        annotation_id = cursor.lastrowid

        study_database.commit()
        cursor.close()

        drawing_thread = threading.Thread(target=draw_polygon, args=(annotation_id,))
        drawing_thread.start()
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error saving to database: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/edit-comment', methods=['POST'])
def annotation_comment():
    try:
        data = request.json
        annotation_id = data['id']
        comment = data['comment']

        cursor = study_database.cursor()

        # Update the comment for the given annotation ID
        cursor.execute("""
            UPDATE annotations
            SET annotation_comment = %s
            WHERE id = %s
        """, (comment, annotation_id))

        study_database.commit()
        cursor.close()

        return jsonify({"success": True, "message": "Comment updated successfully"})

    except Exception as e:
        print(f"Error updating comment: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/save-comment', methods=['POST'])
def save_comment():
    data = request.json
    accession_number = data['accession_number']
    comment = data['comment']

    try:
        cursor = study_database.cursor()

        # Update the comment based on the accession number
        cursor.execute("""
            UPDATE study
            SET study_comment = %s 
            WHERE accession_number = %s
        """, (comment, accession_number))

        study_database.commit()
        cursor.close()

        return jsonify({"success": True})
    except Exception as e:
        print(f"Error saving comment to database: {e}")
        return jsonify({"success": False, "error": str(e)})

def draw_polygon(annotation_id):
    try:
        cursor = study_database.cursor(dictionary=True)
        cursor.execute("USE hip_fracture_study")
        query = "SELECT dir,source_dir, points FROM annotations WHERE id = %s"
        cursor.execute(query, (annotation_id,))
        result = cursor.fetchone()

        if not result:
            raise ValueError(f"No annotation found with ID: {annotation_id}")
        target_dir=result['dir']
        source_dir = result['source_dir']
        points_data = result['points']
        print(target_dir)
        print(source_dir)
        # Load the image using OpenCV
        image = cv2.imread(source_dir)
        if image is None:
            raise ValueError(f"Unable to load image from path: {source_dir}")

        overlay = image.copy()

        # Convert JSON string to Python dictionary (if points data is stored as JSON)
        shapes = json.loads(points_data)  # Use json.loads(points_data) if points_data is JSON formatted string
        if not isinstance(shapes, list):
            raise ValueError("Expected a list of shapes")

            # Convert the list of point dictionaries into a format suitable for OpenCV
        points = np.array([[int(p['x']), int(p['y'])] for p in shapes], np.int32)
        points = points.reshape((-1, 1, 2))  # Reshape for OpenCV
        # Define colors for line and fill (in BGR format)
        line_color = (0, 255, 0)  # Green color for border (B, G, R)
        fill_color = (0, 0, 255)  # Red color for fill (B, G, R)
        alpha = 0.2  # Transparency factor (0 = fully transparent, 1 = fully opaque)

        # Draw filled polygon with transparent fill
        cv2.fillPoly(overlay, [points], fill_color)
        # Draw the border/outline of the polygon
        cv2.polylines(overlay, [points], isClosed=True, color=line_color, thickness=2)

        # Blend the overlay with the original image using the transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Save the annotated image
        cv2.imwrite(target_dir, image)
        print(f"Annotated image saved at: {target_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        study_database.commit()
        cursor.close()

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0',port=5000,debug=True)
    except KeyboardInterrupt:
        study_database.close()
