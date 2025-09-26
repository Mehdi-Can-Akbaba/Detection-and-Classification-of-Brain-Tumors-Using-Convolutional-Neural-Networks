import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri yolu
data_dir = 'Data'

# Parametreler
batch_size = 32
img_height = 224
img_width = 224

# Eğitim veri üretici
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Piksel değerlerini [0, 1] aralığına dönüştür
    rotation_range=40,  # Rastgele döndürme
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    shear_range=0.2,  # Kaydırma
    zoom_range=0.2,  # Yakınlaştırma
    horizontal_flip=True,  # Yatay çevirme
    fill_mode='nearest'  # Boş alanları doldurma
)

# Doğrulama veri üretici
val_datagen = ImageDataGenerator(rescale=1./255)  # Yalnızca normalleştirme

# Eğitim verisini yükle
train_generator = train_datagen.flow_from_directory(
    directory=f"{data_dir}/train",
    target_size=(img_height, img_width),  # Görüntü boyutu
    batch_size=batch_size,
    class_mode='categorical'  # Sınıf modunu belirle
)

# Doğrulama verisini yükle
validation_generator = val_datagen.flow_from_directory(
    directory=f"{data_dir}/val",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Sınıf modunu belirle
)

# Sınıf isimleri
class_indices = train_generator.class_indices
print("Sınıf İsimleri:", class_indices)
