import os
import random

# Klasör yolları
base_dir = 'data2'
sub_dirs = ['train', 'val']
categories = ['Normal', 'Tumor']

def rename_files(base_dir, sub_dirs, categories):
    # Her kategori için numara sayaçları
    category_counters = {category: 1 for category in categories}

    for sub_dir in sub_dirs:
        for category in categories:
            path = os.path.join(base_dir, sub_dir, category)
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")
                continue

            # Dosyaları listele ve karıştır
            files = [f for f in os.listdir(path) if not f.startswith('.')]
            random.shuffle(files)

            for filename in files:
                # Yeni dosya ismi oluştur
                new_name = f"{category} - ({category_counters[category]}){os.path.splitext(filename)[1]}"
                category_counters[category] += 1

                old_file = os.path.join(path, filename)
                new_file = os.path.join(path, new_name)

                # Dosyayı yeniden adlandır
                os.rename(old_file, new_file)
                print(f"Renamed {old_file} to {new_file}")

rename_files(base_dir, sub_dirs, categories)
