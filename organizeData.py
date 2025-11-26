import os
import shutil
import csv

# --- Configuration ---
CLASSIFICATION_FILE = 'classifications.csv'
SOURCE_DIR = 'images'
OUTPUT_DIR = 'output_sorted'
# ---------------------

def sort_images_by_classification():
    # 1. Create the main output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created main output directory: {OUTPUT_DIR}")

    # 2. Read the classification data into a dictionary (filename -> category)
    image_class_map = {}
    try:
        with open(CLASSIFICATION_FILE, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_class_map[row['filename']] = row['classification']
    except FileNotFoundError:
        print(f"Error: Classification file '{CLASSIFICATION_FILE}' not found.")
        return

    # 3. Process each image file in the source directory
    for filename in os.listdir(SOURCE_DIR):
        if filename in image_class_map:
            classification = image_class_map[filename]
            
            # Define the destination path
            destination_folder = os.path.join(OUTPUT_DIR, classification)
            destination_path = os.path.join(destination_folder, filename)
            source_path = os.path.join(SOURCE_DIR, filename)

            # Create the category folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
                print(f"Created category folder: {destination_folder}")

            # Move the file
            try:
                shutil.move(source_path, destination_path)
                print(f"Moved '{filename}' to '{classification}' folder.")
            except shutil.Error as e:
                print(f"Could not move {filename}: {e}")
        else:
            print(f"Warning: '{filename}' not found in classification file. Skipping.")

if __name__ == "__main__":
    sort_images_by_classification()
