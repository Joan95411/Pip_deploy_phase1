import json
import cv2
import numpy as np

# Load the JSON data from a file
with open('test2.json', 'r') as file:  # Replace 'your_json_file.json' with the path to your JSON file
    data = json.load(file)

# Load the image using OpenCV
image_path = 'img3.jpg'
image = cv2.imread(image_path)
# Ensure the image is loaded correctly
if image is None:
    raise ValueError(f"Unable to load image from path: {image_path}")
overlay = image.copy()
# Extract polygon information
shapes = data["shapes"]

# Define colors for line and fill (transparent red) in BGR format
line_color = (0, 255, 0)  # Red color for border (B, G, R)
fill_color = (0, 0, 255)  # Red color for fill (B, G, R)
alpha = 0.2  # Transparency factor (0 = fully transparent, 1 = fully opaque)

for shape in shapes:
    if shape["shape_type"] == "polygon":
        points = np.array(shape["points"], np.int32)  # Convert points to integer NumPy array
        points = points.reshape((-1, 1, 2))  # Reshape for OpenCV
        
        # Draw filled polygon with transparent fill
        cv2.fillPoly(overlay, [points], fill_color)
        # Draw the border/outline of the polygon
        cv2.polylines(overlay, [points], isClosed=True, color=line_color, thickness=2)

# Blend the overlay with the original image using the transparency factor
cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)



# Optionally save the image to a file
cv2.imwrite('image_with_polygons2.jpg', image)  # Change the filename as needed
