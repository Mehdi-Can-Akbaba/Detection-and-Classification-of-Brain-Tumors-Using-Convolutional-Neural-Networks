import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Metrik verilerini yükle
with open('cnn/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Epoch sayısı
epochs = np.array(range(1, len(metrics['train_loss']) + 1))

# Interpolasyon için daha fazla nokta
epochs_new = np.linspace(epochs.min(), epochs.max(), 300)


# 1. Val loss - Epoch ve Train loss - Epoch grafiği
train_loss_spline = make_interp_spline(epochs, metrics['train_loss'], k=3)
train_loss_interpolated = train_loss_spline(epochs_new)

val_loss_spline = make_interp_spline(epochs, metrics['val_loss'], k=3)
val_loss_interpolated = val_loss_spline(epochs_new)

plt.figure(figsize=(10, 5))
plt.plot(epochs_new, train_loss_interpolated, 'r', label='Training loss')
plt.plot(epochs_new, val_loss_interpolated, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()  # Grafiklerin sıkışıklığını azaltmak için
plt.savefig('cnn/loss_plot.png')
plt.show()

# 2. Val Acc - Epoch ve Train Acc - Epoch grafiği
train_acc_spline = make_interp_spline(epochs, metrics['train_acc'], k=3)
train_acc_interpolated = train_acc_spline(epochs_new)

val_acc_spline = make_interp_spline(epochs, metrics['val_acc'], k=3)
val_acc_interpolated = val_acc_spline(epochs_new)

plt.figure(figsize=(10, 5))
plt.plot(epochs_new, train_acc_interpolated, 'r', label='Training Accuracy')
plt.plot(epochs_new, val_acc_interpolated, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()  # Grafiklerin sıkışıklığını azaltmak için
plt.savefig('cnn/accuracy_plot.png')
plt.show()




# Son turdaki ROC verilerini al
fpr = metrics['fpr'][-1]
tpr = metrics['tpr'][-1]
roc_auc = metrics['roc_auc'][-1]

# ROC eğrisini çizme
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Last Epoch')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig('cnn/roc_curve_last_epoch.png')
plt.show()

# ******************
"""
# En yüksek roc_auc değerine sahip indeksi bul
best_index = metrics['roc_auc'].index(max(metrics['roc_auc']))

# En iyi ROC verilerini al
fpr = metrics['fpr'][best_index]
tpr = metrics['tpr'][best_index]
roc_auc = metrics['roc_auc'][best_index]

# ROC eğrisini çizme
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig('swin/swin20/roc_curve_best_epoch.png')
plt.show()
"""
"""
 
# İlk turdaki ROC verilerini al
fpr = metrics['fpr'][0]
tpr = metrics['tpr'][0]
roc_auc = metrics['roc_auc'][0]


# ROC eğrisini çizme
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - First Epoch')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig('levit/roc_curve_first_epoch.png')
plt.show()


# En düşük roc_auc değerine sahip indeksi bul
worst_index = metrics['roc_auc'].index(min(metrics['roc_auc']))

# En kötü ROC verilerini al
fpr = metrics['fpr'][worst_index]
tpr = metrics['tpr'][worst_index]
roc_auc = metrics['roc_auc'][worst_index]

# ROC eğrisini çizme
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Worst Epoch')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig('levit/roc_curve_worst_epoch.png')
plt.show()
"""