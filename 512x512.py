import os
from PIL import Image


def resize_images_in_directory(directory, target_size=(512, 512)):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = Image.open(filepath)
            resized_image = image.resize(target_size, Image.BICUBIC)  # Alternatif olarak BICUBIC veya BILINEAR kullanın
            resized_image.save(filepath)


directory = os.path.join(os.path.dirname(__file__), 'web kazıma/Tumor')
resize_images_in_directory(directory)
