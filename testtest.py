import cv2
import numpy as np
import json
import os

# Sample mock function to simulate a database call (replace with actual DB call in production)
def fetch_annotation_data(annotation_id):
    # Simulating fetched data
    # Replace this with your actual database call
    return {
        'dir': 'D:/Joan/Rad_report_ZGT/SPONGE-BOB.png',  # Path to save the annotated image
        'source_dir': 'D:/Joan/Rad_report_ZGT/Black_Footed_Albatross_0006_796065.jpg',   # Path to the original image
        # 'points': '[{"x": 312.85264458732, "y": 606.751301460524}, {"x": 387.85264458732, "y": 463.751301460524}, '
        #           '{"x": 443.85264458732, "y": 548.751301460524}, {"x": 507.85264458732, "y": 461.751301460524}, '
        #           '{"x": 561.85264458732, "y": 604.751301460524}]'
        'points':'[{"x": 247.8564239, "y": 224.751362495682},'
            '{"x": 322.856423935922, "y": 81.751362495682},'
            '{"x": 378.856423935922, "y": 166.751362495682},'
            '{"x": 442.856423935922, "y": 79.751362495682},'
           ' {"x": 496.856423935922, "y": 222.751362495682}]'
    }

# Function to test drawing polygons
def draw_polygon(annotation_id):
    try:
        # Fetch the annotation data
        result = fetch_annotation_data(annotation_id)

        # Extract the directory paths and points data
        target_dir = result['dir']
        source_dir = result['source_dir']
        points_data = result['points']

        print("Target Directory:", target_dir)
        print("Source Directory:", source_dir)
        print("Points Data (raw):", points_data)

        # Load the image using OpenCV
        image = cv2.imread(source_dir)
        if image is None:
            raise ValueError(f"Unable to load image from path: {source_dir}")

        overlay = image.copy()

        # Parse the points data correctly using json.loads
        shapes = json.loads(points_data)  # Safely parse JSON
        print("Parsed Points Data:", shapes)

        # Check if the points data is a list of dicts
        if not isinstance(shapes, list):
            raise ValueError("Expected a list of shapes")

        # Convert the list of point dictionaries into a format suitable for OpenCV
        points = np.array([[int(p['x']), int(p['y'])] for p in shapes], np.int32)
        points = points.reshape((-1, 1, 2))  # Reshape for OpenCV
        print("Points for OpenCV:", points)

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
        # Log the error internally
        print(f"An error occurred while drawing polygon: {e}")

if __name__=="__main__":
    draw_polygon(1)  # Use an arbitrary annotation ID to fetch data
