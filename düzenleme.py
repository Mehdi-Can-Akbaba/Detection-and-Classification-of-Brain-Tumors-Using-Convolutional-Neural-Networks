import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_folder)
    return images, labels

def preprocess_images(images):
    processed_images = []
    for img in images:
        # Resize image to a fixed size (e.g., 224x224)
        resized_img = cv2.resize(img, (224, 224))
        # Normalize image
        normalized_img = resized_img / 255.0  # Scale pixel values to [0, 1]
        processed_images.append(normalized_img)
    return np.array(processed_images)

def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder.classes_

def main():
    data_dir = 'data2'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Load and preprocess train images
    train_images, train_labels = load_images_from_folder(train_dir)
    train_images = preprocess_images(train_images)

    # Encode train labels
    train_encoded_labels, class_names = encode_labels(train_labels)

    # Load and preprocess validation images
    val_images, val_labels = load_images_from_folder(val_dir)
    val_images = preprocess_images(val_images)

    # Encode validation labels
    val_encoded_labels, _ = encode_labels(val_labels)

    """"
    # Save preprocessed data
    np.savez_compressed('preprocessed_data.npz',
                        train_images=train_images, train_labels=train_encoded_labels,
                        val_images=val_images, val_labels=val_encoded_labels,
                        class_names=class_names)
"""
    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    main()
