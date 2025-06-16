import os
from PIL import Image
from ultralytics import YOLO
import shutil
import argparse

def extract_crops_from_folder(folder_path, model_path, output_base_dir):
    """
    Detects trees in images within a folder using YOLO, and saves each cropped tree
    into a separate folder named after the image.

    Args:
        folder_path (str): Path to the folder containing the images.
        model_path (str): Path to the YOLO model (.pt or .onnx).
        output_base_dir (str): Base directory where cropped images will be saved.
    """
    
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Create the base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Iterate through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(folder_path, filename)
            image_name_without_extension = os.path.splitext(filename)[0]
            
            # Construct the output directory for this image
            output_dir = os.path.join(output_base_dir, image_name_without_extension)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Processing image: {filename}")
            
            # Run YOLO detection
            results = model.predict(
                source=image_path,
                save=False,  # Don't save the annotated image
                save_txt=False, #Don't save txt
                verbose=False #Disable prints during prediction
            )

            # Open the image
            image = Image.open(image_path)
            height, width = image.size

            # Iterate through the detected objects and crop them
            for i, result in enumerate(results):
                boxes = result.boxes.xyxy.tolist()
                for j, box in enumerate(boxes):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Crop the image
                    tree_crop = image.crop((x1, y1, x2, y2))
                    
                    # Save the cropped image
                    crop_filename = f'tree_{j}.jpg'
                    crop_path = os.path.join(output_dir, crop_filename)
                    tree_crop.save(crop_path)
                    print(f"  Saved crop to: {crop_path}")

    print("Finished processing all images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect trees in images and save cropped trees.")
    parser.add_argument("folder_path", help="Path to the folder containing images.")
    parser.add_argument("model_path", help="Path to the YOLO model (.pt or .onnx).")
    parser.add_argument("output_base_dir", help="Base directory to save cropped images.")
    
    args = parser.parse_args()
    
    extract_crops_from_folder(args.folder_path, args.model_path, args.output_base_dir)