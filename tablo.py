import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import torch


output_dir = 'resnet'
os.makedirs(output_dir, exist_ok=True)


with open('resnet/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)


keys_to_include = ['val_loss', 'val_acc', 'f1_score', 'mcc', 'specifity']


filtered_metrics = {key: metrics[key] for key in keys_to_include if key in metrics}


max_length = max(len(filtered_metrics[key]) for key in filtered_metrics.keys())


for key in filtered_metrics.keys():
    while len(filtered_metrics[key]) < max_length:
        filtered_metrics[key].append(0)


table_data = {}


for key in filtered_metrics.keys():
    if isinstance(filtered_metrics[key][0], torch.Tensor):
        table_data[key] = [tensor.item() for tensor in filtered_metrics[key]]
    else:
        table_data[key] = filtered_metrics[key]


df = pd.DataFrame(table_data)

# Virgülden sonra sadece 4 basamağı gösterme fonksiyonu
def format_cell(cell):
    if isinstance(cell, float):
        return f'{cell:.4f}'
    elif isinstance(cell, torch.Tensor):
        return f'{cell.item():.4f}'
    elif isinstance(cell, list) and all(isinstance(item, torch.Tensor) for item in cell):
        return [f'{item.item():.4f}' for item in cell]
    return cell

# Tüm hücreleri formatlamak
for column in df.columns:
    df[column] = df[column].apply(format_cell)

# Tabloyu PNG olarak kaydetme
fig, ax = plt.subplots(figsize=(12, 6))  # Tabloyu çizmek için bir figür oluşturma
ax.axis('tight')
ax.axis('off')

# Tabloyu çizme
ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Tabloyu belirtilen klasöre kaydetme
output_path = os.path.join(output_dir, "metrics_table.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
