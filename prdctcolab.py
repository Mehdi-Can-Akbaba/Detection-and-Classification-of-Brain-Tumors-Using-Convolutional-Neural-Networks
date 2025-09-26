import os
import random
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Model sınıfını tanımlayın
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Veri dönüşümleri
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Modeli yükle
num_classes = 4  # Sınıf sayısı
model = SimpleCNN(num_classes=num_classes)

# CUDA hatasını önlemek için model ağırlıklarını CPU'ya eşleyerek yükleyin
model.load_state_dict(torch.load('cnn100/last_model.pth', map_location=torch.device('cpu')))
model.eval()

# Cihazı ayarla
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tahmin klasörü
output_dir = 'predictions100'
os.makedirs(output_dir, exist_ok=True)

# Sınıf isimleri
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Görselleri seç ve tahmin yap
tahminnum = 2  # Her sınıf için seçilecek görsel sayısı
data_dir = 'Data'

for class_name in class_names:
    class_dir = os.path.join(data_dir, 'train', class_name)

    # Dizin kontrolü
    if not os.path.exists(class_dir):
        print(f"Class directory {class_dir} does not exist.")
        continue

    images = os.listdir(class_dir)
    selected_images = random.sample(images, min(tahminnum, len(images)))  # Seçilen görsel sayısı toplamdan fazla olamaz

    for image_name in selected_images:
        image_path = os.path.join(class_dir, image_name)

        # Görseli aç ve dönüşümü uygula
        try:
            image = Image.open(image_path).convert('RGB')  # Görseli RGB formatında aç
            transform = data_transforms['val']
            image_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Tahmin yap
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            confidence_percentage = confidence[predicted.item()].item()

        # Tahmin sonucunu görselin üzerine yaz
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 14)  # Yazı tipi boyutunu ayarla
        except IOError:
            font = ImageFont.load_default()

        text = f"Predicted: {predicted_class}, Confidence: {confidence_percentage:.2f}%"
        text_position = (10, 10)
        text_color = "yellow"
        draw.text(text_position, text, fill=text_color, font=font)

        # Görseli kaydet
        output_image_path = os.path.join(output_dir, f"{class_name}_{os.path.splitext(image_name)[0]}_prediction.png")
        image.save(output_image_path)

        # Görseli görüntüle (isteğe bağlı)
        # image.show()

print("Tahminler tamamlandı ve görseller kaydedildi.")
