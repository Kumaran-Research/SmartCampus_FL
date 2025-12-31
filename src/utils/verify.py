import cv2
import os

def verify_processed_images(input_dir):
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load: {img_path}")
            elif img.shape != (112, 112):
                print(f"Wrong size: {img_path}, shape: {img.shape}")
            else:
                print(f"Verified: {img_path}")
    print("Verification complete.")

if __name__ == "__main__":
    verify_processed_images('data/client1')
    verify_processed_images('data/client2')