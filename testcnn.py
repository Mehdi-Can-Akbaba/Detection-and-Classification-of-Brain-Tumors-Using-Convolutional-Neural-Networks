import copy
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, \
    matthews_corrcoef, auc
from sklearn.preprocessing import label_binarize
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hiperparametreler
data_dir = 'Data'
batch_size = 32
num_epochs = 30
learning_rate = 0.001

# Veri dönüşümleri
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Veri setlerinin yüklenmesi
image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)  # Sınıf sayısını otomatik olarak ayarla

# Cihazı ayarlama
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CNN modeli tanımlama
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
            nn.Linear(256 * 28 * 28, 1024),  # Görüntü boyutu 224x224 olduğu için 256 * 28 * 28
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Tam bağlantı katmanına bağlamak için girdi boyutunu düzleştir
        x = self.classifier(x)
        return x

# Modeli oluşturma ve cihazda çalıştırma
model = SimpleCNN(num_classes=num_classes).to(device)

# Kayıp fonksiyonu ve optimize edici
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Eğitim fonksiyonu (önceki kodunuzdakiyle aynı)
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    metrics = {
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'fpr': [],
        'tpr': [],
        'roc_auc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                # Eğitim ilerlemesini yüzde olarak gösterme
                if phase == 'train':
                    print('\r', end='')
                    print(f'Epoch {epoch}/{num_epochs - 1} - {((i+1)/len(dataloaders[phase]))*100:.2f}% complete', end='')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')
            epoch_recall = recall_score(all_labels, all_preds, average='macro')
            epoch_precision = precision_score(all_labels, all_preds, average='macro')
            epoch_mcc = matthews_corrcoef(all_labels, all_preds)

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Recall: {epoch_recall:.4f} Precision: {epoch_precision:.4f} MCC: {epoch_mcc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
                all_preds_bin = label_binarize(all_preds, classes=range(num_classes))

                for class_index in range(num_classes):
                    fpr, tpr, _ = roc_curve(all_labels_bin[:, class_index], all_preds_bin[:, class_index])
                    roc_auc = auc(fpr, tpr)
                    metrics['fpr'].append(fpr)
                    metrics['tpr'].append(tpr)
                    metrics['roc_auc'].append(roc_auc)

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        # Her epoch sonunda son modeli kaydet
        torch.save(model.state_dict(), 'cnn/last_model.pth')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    # CNN klasörü
    os.makedirs('cnn', exist_ok=True)

    # Metrikleri kaydet
    with open('cnn/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    # En iyi modeli kaydet
    torch.save(model.state_dict(), 'cnn/best_model.pth')

    return model

if __name__ == "__main__":
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
