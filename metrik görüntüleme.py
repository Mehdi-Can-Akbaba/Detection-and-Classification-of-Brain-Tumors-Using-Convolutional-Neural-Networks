import pickle

# Dosyayı açın
#with open('metrics.pkl', 'rb') as f:
  #  veri = pickle.load(f)

# Veriyi kullanın
#print(veri)

import pickle

# Mevcut metrik dosyasını yükleme
with open('cnn20/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Anahtarlar ve değerlerini yazdırma
for key, value in metrics.items():
    print(f"{key}: {value}")