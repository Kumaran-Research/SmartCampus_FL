import cv2
import numpy as np
import os
from pathlib import Path
import yaml

def preprocess_images(input_dir, output_dir, scale_factor=1.1, min_neighbors=5, target_size=(112, 112)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        output_person_dir = os.path.join(output_dir, person)
        Path(output_person_dir).mkdir(parents=True, exist_ok=True)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if 'pgm' in img_name else cv2.IMREAD_COLOR)
            if img is None:
                continue
            faces = face_cascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, target_size)
                face = face / 255.0
                output_path = os.path.join(output_person_dir, img_name.replace('.pgm', '.jpg'))
                cv2.imwrite(output_path, face * 255)
                break

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    preprocess_images(
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['preprocessing']['scale_factor'],
        config['preprocessing']['min_neighbors'],
        tuple(config['preprocessing']['target_size'])
    )
