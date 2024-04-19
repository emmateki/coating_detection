import os
import cv2

image_folder = "path"
image_folder_cropped = "path"

def crop_images(image_folder,image_folder_cropped):
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            
            height, _, _ = image.shape
            cropped_image = image[:height-120, :, :]
            
            cropped_image_path = os.path.join(image_folder_cropped, filename)
            cv2.imwrite(cropped_image_path, cropped_image)
            
crop_images(image_folder,image_folder_cropped)
