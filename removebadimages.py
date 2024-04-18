# import os

# def delete_small_images(folder_path, size_threshold=30000):  # Set size threshold in bytes
#     """Deletes image files under a specified size threshold from a given folder and its subfolders.

#     Args:
#         folder_path (str): The path to the folder containing images.
#         size_threshold (int, optional): The minimum size (in bytes) to keep an image. Defaults to 30000 (30 KB).
#     """

#     for root, _, files in os.walk(folder_path):
#         for filename in files:
#             file_path = os.path.join(root, filename)
#             if os.path.isfile(file_path):
#                 # Check if file size is less than the threshold
#                 file_size = os.path.getsize(file_path)
#                 if file_size < size_threshold:
#                     # Check if the file extension is an image format (case-insensitive)
#                     if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
#                         print(f"Deleting small image: {file_path} ({file_size} bytes)")
#                         os.remove(file_path)
#                     else:
#                         print(f"Skipping non-image file: {file_path}")

# if __name__ == "__main__":
#     folder_path = r"C:\Users\sim-robinnab\Desktop\Phoenician-Project-Robin_Nabhan\datasets\synthetic_data"  # Replace with the actual path to your folder
#     delete_small_images(folder_path)



import os
import pandas as pd

def delete_small_images(folder_path, size_threshold=30000):
    """Deletes image files under a specified size threshold from a given folder and its subfolders.
    Removes corresponding entries from annotation CSV files.

    Args:
        folder_path (str): The path to the folder containing images.
        size_threshold (int, optional): The minimum size (in bytes) to keep an image. Defaults to 30000 (30 KB).
    """

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                # Check if file size is less than the threshold
                file_size = os.path.getsize(file_path)
                if file_size < size_threshold:
                    # Check if the file extension is an image format (case-insensitive)
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                        print(f"Deleting small image: {file_path} ({file_size} bytes)")
                        os.remove(file_path)
                        
                        # Remove corresponding entry from annotation CSV files
                        remove_annotation_entry(file_path)
                    else:
                        print(f"Skipping non-image file: {file_path}")

def remove_annotation_entry(image_path):
    # Read the annotation CSV file corresponding to the image
    annotation_path = os.path.splitext(image_path)[0] + ".csv"
    if os.path.exists(annotation_path):
        df = pd.read_csv(annotation_path, sep='\t', header=None)
        
        # Remove the row where the first column matches the image path
        df = df[df[0] != image_path]
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(annotation_path, sep='\t', header=None, index=False)
        print(f"Removed annotation entry for: {image_path}")

if __name__ == "__main__":
    folder_path = r"C:\Users\sim-robinnab\Desktop\Phoenician-Project-Robin_Nabhan\datasets\synthetic_data"  # Replace with the actual path to your folder
    delete_small_images(folder_path)
