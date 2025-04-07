import cv2
import numpy as np
import os

def resize_and_pad(img, target_size=(256, 256)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

def process_folder(input_folder, output_folder, target_size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            processed_img = resize_and_pad(img, target_size)
            cv2.imwrite(os.path.join(output_folder, img_name), processed_img)

process_folder("dataset/screenshots", "dataset_resized/screenshots")
process_folder("dataset/screen_photos", "dataset_resized/screen_photos")
