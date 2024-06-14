import os

def create_series_directory(directory: str, image_metadata_list: list):
    # Create the main "studied" directory
    studied_directory = os.path.join(directory, 'studied')
    os.makedirs(studied_directory, exist_ok=True)

    for image_metadata in image_metadata_list:
        # Create the series directory within the "studied" folder
        series_directory = os.path.join(studied_directory, image_metadata.series_instance_uid)
        if not os.path.exists(series_directory):
            os.makedirs(series_directory)
create_series_directory()